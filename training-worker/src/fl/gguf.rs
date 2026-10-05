//! A small GGUF (v2/v3) reader and an F32 writer, enough for LoRA adapters.
//!
//! The round consumes GGUF LoRA adapters (llama.cpp's adapter format: tensors
//! named `<weight>.lora_a` / `<weight>.lora_b`, `general.type = "adapter"`,
//! `adapter.type = "lora"`) and produces one: the start adapter plus the
//! aggregated delta, which llama-server loads with `--lora`.
//!
//! Reading is bounded by the file: every length read from the header is checked
//! against the bytes that remain before anything is allocated, so a hostile
//! header cannot ask for more memory than the file it came in. Only `F32`,
//! `F16` and `BF16` tensors can be read as values; a quantized adapter is
//! refused rather than dequantized by a second implementation.
//!
//! The writer copies the metadata of the adapter it was derived from verbatim
//! (so architecture, alpha and any llama.cpp keys survive) and writes every
//! tensor as `F32`.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// Longest string a header may carry (keys, names, metadata values).
const MAX_STRING: u64 = 1 << 20;
/// Most entries in one metadata array (a 262k-token vocabulary fits).
const MAX_ARRAY: u64 = 1 << 24;
const MAX_KV: u64 = 1 << 16;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_DIMS: u32 = 4;
/// Arrays nest (arrays of arrays); a depth limit keeps recursion bounded.
const MAX_DEPTH: u32 = 4;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_BF16: u32 = 30;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (2 and 3 are supported)")]
    Version(u32),
    #[error("header field {0} is out of bounds for this file")]
    Bounds(&'static str),
    #[error("unknown metadata value type {0}")]
    ValueType(u32),
    #[error("string is not UTF-8")]
    Utf8,
    #[error("tensor {name:?} has type {ty}, only F32, F16 and BF16 can be read")]
    TensorType { name: String, ty: u32 },
    #[error("tensor {0:?} lies outside the file")]
    TensorBounds(String),
    #[error("tensor {0:?} is misaligned")]
    Misaligned(String),
    #[error("duplicate tensor name {0:?}")]
    Duplicate(String),
    #[error("{0}")]
    Invalid(String),
}

/// A metadata value, kept in its declared type so a rewrite is faithful.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// Element type id and the elements.
    Array(u32, Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(..) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dims[0]` is the fastest-moving (row length).
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> Option<u64> {
        self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
    }

    fn byte_len(&self) -> Option<u64> {
        let per = match self.ggml_type {
            GGML_TYPE_F32 => 4,
            GGML_TYPE_F16 | GGML_TYPE_BF16 => 2,
            _ => return None,
        };
        self.elements()?.checked_mul(per)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf {
    pub version: u32,
    pub kv: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute file offset of the data section.
    pub data_offset: u64,
    pub file_len: u64,
}

impl Gguf {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// True when the metadata says this is a llama.cpp LoRA adapter.
    pub fn is_lora_adapter(&self) -> bool {
        self.get("general.type").and_then(Value::as_str) == Some("adapter")
            && self.get("adapter.type").and_then(Value::as_str) == Some("lora")
    }
}

struct Cursor<R> {
    r: R,
    pos: u64,
    len: u64,
}

impl<R: Read> Cursor<R> {
    fn take(&mut self, n: u64, what: &'static str) -> Result<Vec<u8>, GgufError> {
        if n > self.len.saturating_sub(self.pos) {
            return Err(GgufError::Bounds(what));
        }
        let n_usize = usize::try_from(n).map_err(|_| GgufError::Bounds(what))?;
        let mut buf = vec![0u8; n_usize];
        self.r.read_exact(&mut buf)?;
        self.pos += n;
        Ok(buf)
    }

    fn arr<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], GgufError> {
        let v = self.take(N as u64, what)?;
        let mut a = [0u8; N];
        a.copy_from_slice(&v);
        Ok(a)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.arr(what)?))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.arr(what)?))
    }

    fn string(&mut self, what: &'static str) -> Result<String, GgufError> {
        let n = self.u64(what)?;
        if n > MAX_STRING {
            return Err(GgufError::Bounds(what));
        }
        String::from_utf8(self.take(n, what)?).map_err(|_| GgufError::Utf8)
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, GgufError> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>("u8")?[0]),
            1 => Value::I8(i8::from_le_bytes(self.arr("i8")?)),
            2 => Value::U16(u16::from_le_bytes(self.arr("u16")?)),
            3 => Value::I16(i16::from_le_bytes(self.arr("i16")?)),
            4 => Value::U32(self.u32("u32")?),
            5 => Value::I32(i32::from_le_bytes(self.arr("i32")?)),
            6 => Value::F32(f32::from_le_bytes(self.arr("f32")?)),
            7 => Value::Bool(self.arr::<1>("bool")?[0] != 0),
            8 => Value::Str(self.string("string value")?),
            9 => {
                if depth >= MAX_DEPTH {
                    return Err(GgufError::Bounds("array nesting"));
                }
                let ety = self.u32("array type")?;
                let n = self.u64("array length")?;
                // Every element is at least one byte, so a count beyond the
                // remaining file is a lie told to make us allocate.
                if n > MAX_ARRAY || n > self.len.saturating_sub(self.pos) {
                    return Err(GgufError::Bounds("array length"));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(ety, depth + 1)?);
                }
                Value::Array(ety, v)
            }
            10 => Value::U64(self.u64("u64")?),
            11 => Value::I64(i64::from_le_bytes(self.arr("i64")?)),
            12 => Value::F64(f64::from_le_bytes(self.arr("f64")?)),
            other => return Err(GgufError::ValueType(other)),
        })
    }
}

/// Read the header (metadata and tensor table) without reading tensor data.
pub fn read_header<R: Read + Seek>(mut r: R) -> Result<Gguf, GgufError> {
    let len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let mut c = Cursor { r, pos: 0, len };
    if c.arr::<4>("magic")? != MAGIC {
        return Err(GgufError::BadMagic);
    }
    let version = c.u32("version")?;
    if version != 2 && version != 3 {
        return Err(GgufError::Version(version));
    }
    let n_tensors = c.u64("tensor count")?;
    let n_kv = c.u64("kv count")?;
    if n_tensors > MAX_TENSORS {
        return Err(GgufError::Bounds("tensor count"));
    }
    if n_kv > MAX_KV {
        return Err(GgufError::Bounds("kv count"));
    }
    let mut kv = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = c.string("key")?;
        let ty = c.u32("value type")?;
        let v = c.value(ty, 0)?;
        kv.push((key, v));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    let mut names = std::collections::BTreeSet::new();
    for _ in 0..n_tensors {
        let name = c.string("tensor name")?;
        let nd = c.u32("tensor dims")?;
        if nd == 0 || nd > MAX_DIMS {
            return Err(GgufError::Bounds("tensor dims"));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(c.u64("tensor dim")?);
        }
        let ggml_type = c.u32("tensor type")?;
        let offset = c.u64("tensor offset")?;
        if !names.insert(name.clone()) {
            return Err(GgufError::Duplicate(name));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    let alignment = match kv.iter().find(|(k, _)| k == "general.alignment") {
        Some((_, Value::U32(a))) if *a > 0 && a.is_power_of_two() => u64::from(*a),
        Some(_) => return Err(GgufError::Invalid("general.alignment is invalid".into())),
        None => DEFAULT_ALIGNMENT,
    };
    let data_offset = c.pos.div_ceil(alignment) * alignment;
    if data_offset > len {
        return Err(GgufError::Bounds("data section"));
    }
    for t in &tensors {
        if t.offset % alignment != 0 {
            return Err(GgufError::Misaligned(t.name.clone()));
        }
    }
    Ok(Gguf {
        version,
        kv,
        tensors,
        alignment,
        data_offset,
        file_len: len,
    })
}

/// Read a whole file's header.
pub fn read_header_path(path: &Path) -> Result<Gguf, GgufError> {
    read_header(std::io::BufReader::new(std::fs::File::open(path)?))
}

/// IEEE half to single, exact (every f16 is representable as an f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalise the mantissa.
            let mut e: i32 = -14;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13) | 0x0040_0000,
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Read one tensor's values as `f32`.
pub fn read_tensor_f32<R: Read + Seek>(
    mut r: R,
    g: &Gguf,
    t: &TensorInfo,
) -> Result<Vec<f32>, GgufError> {
    let bytes = t.byte_len().ok_or_else(|| GgufError::TensorType {
        name: t.name.clone(),
        ty: t.ggml_type,
    })?;
    let start = g
        .data_offset
        .checked_add(t.offset)
        .ok_or_else(|| GgufError::TensorBounds(t.name.clone()))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| GgufError::TensorBounds(t.name.clone()))?;
    if end > g.file_len {
        return Err(GgufError::TensorBounds(t.name.clone()));
    }
    r.seek(SeekFrom::Start(start))?;
    let n = usize::try_from(bytes).map_err(|_| GgufError::TensorBounds(t.name.clone()))?;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(match t.ggml_type {
        GGML_TYPE_F32 => buf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GGML_TYPE_F16 => buf
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        // BF16 is the top half of an f32.
        _ => buf
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect(),
    })
}

/// A LoRA adapter loaded into memory: header plus every tensor as `f32`, in the
/// file's tensor order.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub header: Gguf,
    pub values: Vec<Vec<f32>>,
}

impl Adapter {
    /// Load an adapter, refusing anything that is not a LoRA adapter or holds a
    /// tensor this reader cannot represent exactly.
    pub fn load(path: &Path) -> Result<Self, GgufError> {
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(f);
        let header = read_header(&mut r)?;
        if !header.is_lora_adapter() {
            return Err(GgufError::Invalid(format!(
                "{} is not a LoRA adapter (general.type must be \"adapter\" and \
                 adapter.type \"lora\")",
                path.display()
            )));
        }
        let mut values = Vec::with_capacity(header.tensors.len());
        for t in &header.tensors {
            if !(t.name.ends_with(".lora_a") || t.name.ends_with(".lora_b")) {
                return Err(GgufError::Invalid(format!(
                    "adapter tensor {:?} is neither a lora_a nor a lora_b",
                    t.name
                )));
            }
            values.push(read_tensor_f32(&mut r, &header, t)?);
        }
        Ok(Self { header, values })
    }
}

fn put_str<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn put_value<W: Write>(w: &mut W, v: &Value) -> std::io::Result<()> {
    match v {
        Value::U8(x) => w.write_all(&[*x]),
        Value::I8(x) => w.write_all(&x.to_le_bytes()),
        Value::U16(x) => w.write_all(&x.to_le_bytes()),
        Value::I16(x) => w.write_all(&x.to_le_bytes()),
        Value::U32(x) => w.write_all(&x.to_le_bytes()),
        Value::I32(x) => w.write_all(&x.to_le_bytes()),
        Value::F32(x) => w.write_all(&x.to_le_bytes()),
        Value::Bool(x) => w.write_all(&[u8::from(*x)]),
        Value::Str(s) => put_str(w, s),
        Value::Array(ety, items) => {
            w.write_all(&ety.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for i in items {
                put_value(w, i)?;
            }
            Ok(())
        }
        Value::U64(x) => w.write_all(&x.to_le_bytes()),
        Value::I64(x) => w.write_all(&x.to_le_bytes()),
        Value::F64(x) => w.write_all(&x.to_le_bytes()),
    }
}

/// One tensor to write: name, GGML dims and `f32` values.
pub struct OutTensor<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub values: &'a [f32],
}

/// Serialize a GGUF v3 file with `F32` tensors. `kv` is written verbatim;
/// alignment follows its `general.alignment` when present.
pub fn encode_f32(kv: &[(String, Value)], tensors: &[OutTensor<'_>]) -> Result<Vec<u8>, GgufError> {
    let alignment = match kv.iter().find(|(k, _)| k == "general.alignment") {
        Some((_, Value::U32(a))) if *a > 0 && a.is_power_of_two() => u64::from(*a),
        Some(_) => return Err(GgufError::Invalid("general.alignment is invalid".into())),
        None => DEFAULT_ALIGNMENT,
    };
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        put_str(&mut out, k)?;
        out.extend_from_slice(&v.type_id().to_le_bytes());
        put_value(&mut out, v)?;
    }
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        let n: u64 = t.dims.iter().product();
        if n != t.values.len() as u64 {
            return Err(GgufError::Invalid(format!(
                "tensor {:?}: dims hold {n} values, got {}",
                t.name,
                t.values.len()
            )));
        }
        put_str(&mut out, t.name)?;
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offsets.push(offset);
        offset = (offset + n * 4).div_ceil(alignment) * alignment;
    }
    let data_start = (out.len() as u64).div_ceil(alignment) * alignment;
    out.resize(data_start as usize, 0);
    for (t, off) in tensors.iter().zip(offsets) {
        out.resize((data_start + off) as usize, 0);
        for v in t.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    include!("gguf_tests.rs");
}
