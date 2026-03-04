//! Load SMPL model directly from .npz files.
//!
//! NPZ = ZIP archive of .npy files. Each .npy file has:
//!   - Magic: \x93NUMPY (6 bytes)
//!   - Version: major (1 byte) + minor (1 byte)
//!   - Header length: u16 LE (v1) or u32 LE (v2)
//!   - Header: Python dict literal with 'descr', 'fortran_order', 'shape'
//!   - Data: raw bytes
//!
//! The SMPL .npz contains standard numeric arrays (float64, uint32).
//! We parse them, cast to f32, and precompute derived arrays
//! (j_template, j_shapedirs, sparse weights, etc.) — the same
//! precomputation that `export_model_bin.py` does offline.

use std::io::{Cursor, Read};

use crate::model_data::SmplModelData;
use crate::sparse_lbs;

#[derive(Debug)]
pub enum NpzError {
    Zip(String),
    Npy(String),
    MissingArray(&'static str),
    ShapeMismatch { name: &'static str, expected: &'static str, got: String },
}

impl std::fmt::Display for NpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NpzError::Zip(e) => write!(f, "ZIP error: {}", e),
            NpzError::Npy(e) => write!(f, "NPY parse error: {}", e),
            NpzError::MissingArray(name) => write!(f, "Missing array '{}' in npz", name),
            NpzError::ShapeMismatch { name, expected, got } =>
                write!(f, "Array '{}' shape mismatch: expected {}, got {}", name, expected, got),
        }
    }
}

impl std::error::Error for NpzError {}

/// Parsed NPY array metadata.
struct NpyArray {
    shape: Vec<usize>,
    dtype: String,
    fortran_order: bool,
    data: Vec<u8>,
}

/// Parse NPY header from a reader.
fn parse_npy<R: Read>(reader: &mut R) -> Result<NpyArray, NpzError> {
    // Magic: \x93NUMPY
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).map_err(|e| NpzError::Npy(format!("read magic: {}", e)))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(NpzError::Npy("bad magic".into()));
    }

    // Version
    let mut ver = [0u8; 2];
    reader.read_exact(&mut ver).map_err(|e| NpzError::Npy(format!("read version: {}", e)))?;
    let major = ver[0];

    // Header length
    let header_len = if major <= 1 {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf).map_err(|e| NpzError::Npy(format!("read header len: {}", e)))?;
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(|e| NpzError::Npy(format!("read header len v2: {}", e)))?;
        u32::from_le_bytes(buf) as usize
    };

    // Header string (Python dict literal)
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes).map_err(|e| NpzError::Npy(format!("read header: {}", e)))?;
    let header = String::from_utf8_lossy(&header_bytes);

    // Parse header dict fields
    let dtype = extract_field(&header, "descr")
        .ok_or_else(|| NpzError::Npy("missing 'descr' in header".into()))?;
    let fortran_str = extract_field(&header, "fortran_order")
        .unwrap_or_else(|| "False".into());
    let fortran_order = fortran_str.contains("True");
    let shape = extract_shape(&header)
        .ok_or_else(|| NpzError::Npy("missing 'shape' in header".into()))?;

    // Read remaining data
    let num_elements: usize = shape.iter().product::<usize>().max(1);
    let element_size = dtype_size(&dtype)?;
    let data_size = num_elements * element_size;
    let mut data = vec![0u8; data_size];
    reader.read_exact(&mut data).map_err(|e| NpzError::Npy(format!("read data ({} bytes): {}", data_size, e)))?;

    Ok(NpyArray { shape, dtype, fortran_order, data })
}

/// Extract a string field from NPY header dict.
/// Header looks like: {'descr': '<f8', 'fortran_order': False, 'shape': (6890, 3), }
fn extract_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{}':", field);
    let start = header.find(&pattern)? + pattern.len();
    let rest = header[start..].trim_start();

    if rest.starts_with('\'') || rest.starts_with('"') {
        // String value
        let quote = rest.as_bytes()[0] as char;
        let end = rest[1..].find(quote)? + 1;
        Some(rest[1..end].to_string())
    } else {
        // Non-string value (True/False/number)
        let end = rest.find([',', '}'].as_ref()).unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }
}

/// Extract shape tuple from NPY header.
fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")? + "'shape':".len();
    let rest = header[start..].trim_start();
    let paren_start = rest.find('(')?;
    let paren_end = rest.find(')')?;
    let inner = &rest[paren_start + 1..paren_end];

    if inner.trim().is_empty() {
        return Some(vec![]); // scalar
    }

    let dims: Result<Vec<usize>, _> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();

    dims.ok()
}

/// Get element size in bytes from dtype string.
fn dtype_size(dtype: &str) -> Result<usize, NpzError> {
    // Common dtypes: <f8 (float64), <f4 (float32), <u4 (uint32), <i4 (int32), <U3 (unicode)
    let clean = dtype.trim_start_matches(['<', '>', '|', '=']);
    match clean.chars().next() {
        Some('f') => {
            let n: usize = clean[1..].parse().map_err(|_| NpzError::Npy(format!("bad dtype: {}", dtype)))?;
            Ok(n)
        }
        Some('u') | Some('i') => {
            let n: usize = clean[1..].parse().map_err(|_| NpzError::Npy(format!("bad dtype: {}", dtype)))?;
            Ok(n)
        }
        Some('U') => {
            // Unicode string: 4 bytes per character
            let n: usize = clean[1..].parse().map_err(|_| NpzError::Npy(format!("bad dtype: {}", dtype)))?;
            Ok(n * 4)
        }
        Some('b') => Ok(1), // bool
        _ => Err(NpzError::Npy(format!("unsupported dtype: {}", dtype))),
    }
}

impl NpyArray {
    /// Convert array data to Vec<f32>, handling float64→float32 casting.
    fn to_f32(&self) -> Result<Vec<f32>, NpzError> {
        let n: usize = self.shape.iter().product::<usize>().max(1);

        if self.fortran_order && self.shape.len() > 1 {
            // Fortran (column-major) → need to transpose to C order
            return self.to_f32_transpose();
        }

        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);
        match clean_dtype {
            "f8" => {
                // float64 → float32
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                    result.push(f64::from_le_bytes(bytes) as f32);
                }
                Ok(result)
            }
            "f4" => {
                // float32 direct
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to f32", self.dtype))),
        }
    }

    /// Convert Fortran-order array to C-order f32.
    fn to_f32_transpose(&self) -> Result<Vec<f32>, NpzError> {
        let shape = &self.shape;
        let ndim = shape.len();
        let n: usize = shape.iter().product();
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);

        // Read raw values
        let raw: Vec<f64> = match clean_dtype {
            "f8" => (0..n).map(|i| {
                let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                f64::from_le_bytes(bytes)
            }).collect(),
            "f4" => (0..n).map(|i| {
                let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                f32::from_le_bytes(bytes) as f64
            }).collect(),
            _ => return Err(NpzError::Npy(format!("cannot convert Fortran dtype '{}' to f32", self.dtype))),
        };

        // Compute strides for Fortran order
        let mut f_strides = vec![1usize; ndim];
        for i in 1..ndim {
            f_strides[i] = f_strides[i - 1] * shape[i - 1];
        }

        // Compute strides for C order
        let mut c_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            c_strides[i] = c_strides[i + 1] * shape[i + 1];
        }

        let mut result = vec![0.0f32; n];
        let mut indices = vec![0usize; ndim];

        for c_idx in 0..n {
            // Compute multi-index from C-order flat index
            let mut remainder = c_idx;
            for d in 0..ndim {
                indices[d] = remainder / c_strides[d];
                remainder %= c_strides[d];
            }

            // Compute Fortran-order flat index
            let f_idx: usize = indices.iter().zip(f_strides.iter()).map(|(&i, &s)| i * s).sum();
            result[c_idx] = raw[f_idx] as f32;
        }

        Ok(result)
    }

    /// Convert array to Vec<u32>.
    fn to_u32(&self) -> Result<Vec<u32>, NpzError> {
        let n: usize = self.shape.iter().product::<usize>().max(1);
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);

        match clean_dtype {
            "u4" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    result.push(u32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            "i4" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    result.push(i32::from_le_bytes(bytes) as u32);
                }
                Ok(result)
            }
            "i8" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                    result.push(i64::from_le_bytes(bytes) as u32);
                }
                Ok(result)
            }
            "u8" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                    result.push(u64::from_le_bytes(bytes) as u32);
                }
                Ok(result)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to u32", self.dtype))),
        }
    }

    /// Convert to Vec<i32>.
    fn to_i32(&self) -> Result<Vec<i32>, NpzError> {
        let n: usize = self.shape.iter().product::<usize>().max(1);
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);

        match clean_dtype {
            "u4" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    result.push(u32::from_le_bytes(bytes) as i32);
                }
                Ok(result)
            }
            "i4" => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    result.push(i32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to i32", self.dtype))),
        }
    }
}

/// Load an SMPL model directly from .npz bytes.
///
/// Performs all precomputation at load time:
/// - j_template = J_regressor @ v_template
/// - j_shapedirs = einsum("jv,vds->jds", J_regressor, shapedirs)
/// - sparse weight extraction (top-4)
/// - rest_pose_y_offset = -min(v_template[:, 1])
pub fn load_npz(data: &[u8]) -> Result<SmplModelData, NpzError> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| NpzError::Zip(format!("{}", e)))?;

    // Read required arrays
    let v_template_arr = read_npy_from_zip(&mut archive, "v_template")?;
    let shapedirs_arr = read_npy_from_zip(&mut archive, "shapedirs")?;
    let posedirs_arr = read_npy_from_zip(&mut archive, "posedirs")?;
    let j_regressor_arr = read_npy_from_zip(&mut archive, "J_regressor")?;
    let weights_arr = read_npy_from_zip(&mut archive, "weights")?;
    let faces_arr = read_npy_from_zip(&mut archive, "f")?;
    let kintree_arr = read_npy_from_zip(&mut archive, "kintree_table")?;

    // Validate shapes
    let num_vertices = v_template_arr.shape[0];
    let num_joints = j_regressor_arr.shape[0];
    let num_shape_params = shapedirs_arr.shape[2];

    if v_template_arr.shape.len() != 2 || v_template_arr.shape[1] != 3 {
        return Err(NpzError::ShapeMismatch {
            name: "v_template", expected: "[V, 3]", got: format!("{:?}", v_template_arr.shape),
        });
    }
    if j_regressor_arr.shape.len() != 2 || j_regressor_arr.shape[1] != num_vertices {
        return Err(NpzError::ShapeMismatch {
            name: "J_regressor", expected: "[J, V]", got: format!("{:?}", j_regressor_arr.shape),
        });
    }

    let v = num_vertices;
    let j = num_joints;
    let s = num_shape_params;

    // Determine pose params
    let p = if posedirs_arr.shape.len() == 3 {
        posedirs_arr.shape[2]
    } else if posedirs_arr.shape.len() == 2 {
        posedirs_arr.shape[0]
    } else {
        return Err(NpzError::ShapeMismatch {
            name: "posedirs", expected: "[V,3,P] or [P,V*3]", got: format!("{:?}", posedirs_arr.shape),
        });
    };

    // Convert to f32
    let v_template = v_template_arr.to_f32()?;
    let j_regressor = j_regressor_arr.to_f32()?;
    let lbs_weights = weights_arr.to_f32()?;
    let faces = faces_arr.to_u32()?;

    // shapedirs: ensure [V, 3, S] layout
    let shapedirs = shapedirs_arr.to_f32()?;
    assert_eq!(shapedirs.len(), v * 3 * s);

    // posedirs: handle both [V, 3, P] and [P, V*3] layouts
    let posedirs = if posedirs_arr.shape.len() == 3 && posedirs_arr.shape[0] == v {
        // Already [V, 3, P]
        posedirs_arr.to_f32()?
    } else if posedirs_arr.shape.len() == 2 {
        // [P, V*3] → transpose to [V*3, P] then reshape to [V, 3, P]
        let raw = posedirs_arr.to_f32()?;
        let rows = posedirs_arr.shape[0]; // P
        let cols = posedirs_arr.shape[1]; // V*3
        // raw is row-major [P, V*3], we need [V, 3, P] = [V*3, P] transposed read
        let mut result = vec![0.0f32; v * 3 * p];
        for vi_d in 0..cols { // V*3
            for pi in 0..rows { // P
                result[vi_d * p + pi] = raw[pi * cols + vi_d];
            }
        }
        result
    } else {
        return Err(NpzError::ShapeMismatch {
            name: "posedirs", expected: "[V,3,P] or [P,V*3]", got: format!("{:?}", posedirs_arr.shape),
        });
    };

    // Parents from kintree_table[0]
    let kintree = kintree_arr.to_i32()?;
    // kintree_table is [2, J], we need row 0
    let parents: Vec<i32> = (0..j).map(|ji| kintree[ji]).collect();

    // === Precomputation ===

    // j_template = J_regressor @ v_template  [J, 3]
    let mut j_template = vec![0.0f32; j * 3];
    for ji in 0..j {
        for d in 0..3 {
            let mut sum = 0.0f32;
            for vi in 0..v {
                sum += j_regressor[ji * v + vi] * v_template[vi * 3 + d];
            }
            j_template[ji * 3 + d] = sum;
        }
    }

    // j_shapedirs = einsum("jv,vds->jds", J_regressor, shapedirs)  [J, 3, S]
    // shapedirs layout: [V, 3, S] = [vi*3*S + d*S + si]
    let mut j_shapedirs = vec![0.0f32; j * 3 * s];
    for ji in 0..j {
        for d in 0..3 {
            for si in 0..s {
                let mut sum = 0.0f32;
                for vi in 0..v {
                    sum += j_regressor[ji * v + vi] * shapedirs[vi * 3 * s + d * s + si];
                }
                j_shapedirs[ji * 3 * s + d * s + si] = sum;
            }
        }
    }

    // rest_pose_y_offset = -min(v_template[:, 1])
    let mut min_y = f32::INFINITY;
    for vi in 0..v {
        let y = v_template[vi * 3 + 1];
        if y < min_y { min_y = y; }
    }
    let rest_pose_y_offset = -min_y;

    // Sparse weights (top-4)
    let nnz = 4;
    let (sparse_indices, sparse_weights) = sparse_lbs::extract_sparse_weights(
        &lbs_weights, v, j, nnz,
    );

    let num_faces = faces.len() / 3;

    Ok(SmplModelData {
        num_vertices: v,
        num_joints: j,
        num_shape_params: s,
        num_pose_params: p,
        nnz_per_vertex: nnz,
        num_faces,
        rest_pose_y_offset,
        v_template,
        shapedirs,
        posedirs,
        j_template,
        j_shapedirs,
        parents,
        lbs_weights,
        sparse_indices,
        sparse_weights,
        faces,
    })
}

/// Read a .npy file from inside a ZIP archive.
fn read_npy_from_zip(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    name: &'static str,
) -> Result<NpyArray, NpzError> {
    // Try "name.npy" first, fall back to "name"
    let file_name = format!("{}.npy", name);
    let has_npy = archive.by_name(&file_name).is_ok();
    let actual_name = if has_npy { file_name.as_str() } else { name };
    let mut file = archive.by_name(actual_name)
        .map_err(|_| NpzError::MissingArray(name))?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| NpzError::Zip(format!("read {}: {}", name, e)))?;

    let mut cursor = Cursor::new(buf.as_slice());
    parse_npy(&mut cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_field() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (6890, 3), }";
        assert_eq!(extract_field(header, "descr"), Some("<f8".into()));
        assert_eq!(extract_field(header, "fortran_order"), Some("False".into()));
    }

    #[test]
    fn test_extract_shape() {
        assert_eq!(extract_shape("'shape': (6890, 3), "), Some(vec![6890, 3]));
        assert_eq!(extract_shape("'shape': (24,), "), Some(vec![24]));
        assert_eq!(extract_shape("'shape': (), "), Some(vec![]));
        assert_eq!(extract_shape("'shape': (6890, 3, 300), "), Some(vec![6890, 3, 300]));
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size("<f8").unwrap(), 8);
        assert_eq!(dtype_size("<f4").unwrap(), 4);
        assert_eq!(dtype_size("<u4").unwrap(), 4);
        assert_eq!(dtype_size("<U3").unwrap(), 12);
    }
}
