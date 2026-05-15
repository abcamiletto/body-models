//! Binary model data format deserialization.
//!
//! Binary layout:
//!   magic: u32 (0x534D504C = "SMPL")
//!   version: u32 (1)
//!   num_vertices: u32
//!   num_joints: u32
//!   num_shape_params: u32
//!   num_pose_params: u32  (= (num_joints - 1) * 9)
//!   nnz_per_vertex: u32   (= 4 for SMPL)
//!   num_faces: u32
//!   rest_pose_y_offset: f32
//!   padding: u32 (reserved)
//!   --- arrays, each prefixed by u64 byte-length ---
//!   v_template:      [V, 3] f32
//!   shapedirs:       [V, 3, S] f32
//!   posedirs:        [V, 3, P] f32   (P = (J-1)*9, laid out per-vertex)
//!   j_template:      [J, 3] f32
//!   j_shapedirs:     [J, 3, S] f32
//!   parents:         [J] i32
//!   lbs_weights:     [V, J] f32      (dense, for CPU fallback)
//!   sparse_indices:  [V, K] u32      (K = nnz_per_vertex)
//!   sparse_weights:  [V, K] f32
//!   faces:           [F, 3] u32

const MAGIC: u32 = 0x534D504C; // "SMPL"
const VERSION: u32 = 1;

/// Parsed SMPL model data ready for forward pass computation.
pub struct SmplModelData {
    pub num_vertices: usize,
    pub num_joints: usize,
    pub num_shape_params: usize,
    pub num_pose_params: usize,
    pub nnz_per_vertex: usize,
    pub num_faces: usize,
    pub rest_pose_y_offset: f32,

    /// Template vertices [V*3], row-major
    pub v_template: Vec<f32>,
    /// Shape blend shapes [V*3*S], indexed as [v*3*S + d*S + s]
    pub shapedirs: Vec<f32>,
    /// Pose blend shapes [V*3*P], indexed as [v*3*P + d*P + p]
    pub posedirs: Vec<f32>,
    /// Precomputed joint template positions [J*3]
    pub j_template: Vec<f32>,
    /// Precomputed joint shape directions [J*3*S]
    pub j_shapedirs: Vec<f32>,
    /// Parent joint indices [J], parents[0] = -1
    pub parents: Vec<i32>,
    /// Dense LBS weights [V*J] for CPU path
    pub lbs_weights: Vec<f32>,
    /// Sparse joint indices [V*K]
    pub sparse_indices: Vec<u32>,
    /// Sparse joint weights [V*K]
    pub sparse_weights: Vec<f32>,
    /// Triangle faces [F*3]
    pub faces: Vec<u32>,
}

#[derive(Debug)]
pub enum ParseError {
    TooShort,
    BadMagic(u32),
    BadVersion(u32),
    ArraySizeMismatch { name: &'static str, expected: usize, got: usize },
    UnexpectedTrailing(usize),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::TooShort => write!(f, "Binary data too short for header"),
            ParseError::BadMagic(m) => write!(f, "Bad magic: 0x{:08X}, expected 0x{:08X}", m, MAGIC),
            ParseError::BadVersion(v) => write!(f, "Bad version: {}, expected {}", v, VERSION),
            ParseError::ArraySizeMismatch { name, expected, got } =>
                write!(f, "Array '{}' size mismatch: expected {} bytes, got {}", name, expected, got),
            ParseError::UnexpectedTrailing(n) =>
                write!(f, "Unexpected {} trailing bytes", n),
        }
    }
}

impl std::error::Error for ParseError {}

/// Read a little-endian u32 from bytes.
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, ParseError> {
    if *offset + 4 > data.len() {
        return Err(ParseError::TooShort);
    }
    let val = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

/// Read a little-endian f32 from bytes.
fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32, ParseError> {
    if *offset + 4 > data.len() {
        return Err(ParseError::TooShort);
    }
    let val = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

/// Read a length-prefixed Pod array via bulk memcpy.
fn read_pod_array<T: bytemuck::Pod>(data: &[u8], offset: &mut usize, name: &'static str, expected_count: usize) -> Result<Vec<T>, ParseError> {
    if *offset + 8 > data.len() {
        return Err(ParseError::TooShort);
    }
    let byte_len = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;

    let expected_bytes = expected_count * std::mem::size_of::<T>();
    if byte_len != expected_bytes {
        return Err(ParseError::ArraySizeMismatch { name, expected: expected_bytes, got: byte_len });
    }
    if *offset + byte_len > data.len() {
        return Err(ParseError::TooShort);
    }

    let mut arr = vec![T::zeroed(); expected_count];
    bytemuck::cast_slice_mut::<T, u8>(&mut arr).copy_from_slice(&data[*offset..*offset + byte_len]);
    *offset += byte_len;
    Ok(arr)
}

impl SmplModelData {
    /// Parse binary model data.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ParseError> {
        let mut offset = 0;

        // Header
        let magic = read_u32(data, &mut offset)?;
        if magic != MAGIC {
            return Err(ParseError::BadMagic(magic));
        }
        let version = read_u32(data, &mut offset)?;
        if version != VERSION {
            return Err(ParseError::BadVersion(version));
        }

        let num_vertices = read_u32(data, &mut offset)? as usize;
        let num_joints = read_u32(data, &mut offset)? as usize;
        let num_shape_params = read_u32(data, &mut offset)? as usize;
        let num_pose_params = read_u32(data, &mut offset)? as usize;
        let nnz_per_vertex = read_u32(data, &mut offset)? as usize;
        let num_faces = read_u32(data, &mut offset)? as usize;
        let rest_pose_y_offset = read_f32(data, &mut offset)?;
        let _padding = read_u32(data, &mut offset)?;

        let v = num_vertices;
        let j = num_joints;
        let s = num_shape_params;
        let p = num_pose_params;
        let k = nnz_per_vertex;
        let f = num_faces;

        // Arrays
        let v_template = read_pod_array::<f32>(data, &mut offset, "v_template", v * 3)?;
        let shapedirs = read_pod_array::<f32>(data, &mut offset, "shapedirs", v * 3 * s)?;
        let posedirs = read_pod_array::<f32>(data, &mut offset, "posedirs", v * 3 * p)?;
        let j_template = read_pod_array::<f32>(data, &mut offset, "j_template", j * 3)?;
        let j_shapedirs = read_pod_array::<f32>(data, &mut offset, "j_shapedirs", j * 3 * s)?;
        let parents = read_pod_array::<i32>(data, &mut offset, "parents", j)?;
        let lbs_weights = read_pod_array::<f32>(data, &mut offset, "lbs_weights", v * j)?;
        let sparse_indices = read_pod_array::<u32>(data, &mut offset, "sparse_indices", v * k)?;
        let sparse_weights = read_pod_array::<f32>(data, &mut offset, "sparse_weights", v * k)?;
        let faces = read_pod_array::<u32>(data, &mut offset, "faces", f * 3)?;

        if offset != data.len() {
            return Err(ParseError::UnexpectedTrailing(data.len() - offset));
        }

        Ok(SmplModelData {
            num_vertices,
            num_joints,
            num_shape_params,
            num_pose_params,
            nnz_per_vertex,
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
}
