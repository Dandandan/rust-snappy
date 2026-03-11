use std::ptr;

use crate::bytes;
use crate::error::{Error, Result};
use crate::tag;
use crate::MAX_INPUT_SIZE;

/// A lookup table for quickly computing the various attributes derived from a
/// tag byte.
const TAG_LOOKUP_TABLE: TagLookupTable = TagLookupTable(tag::TAG_LOOKUP_TABLE);

/// Combined length-and-offset table inspired by C++ snappy's kLengthMinusOffset.
///
/// For each tag byte, encodes (as i16):
///   - Short literals (len 1-16): `length - 256` (always negative)
///   - Copy-1: `length - (offset_high_bits << 8)`
///   - Copy-2 (len 1-16): `length`
///   - Extended literals, long copies, copy-4: flag value (0x0180)
///
/// The key property: for common tags (short literals and copies where
/// offset > length), `entry <= ExtractOffset(trailing_bytes, tag_type)`,
/// allowing a single unified code path without branching on tag type.
const LENGTH_MINUS_OFFSET: [i16; 256] = make_length_minus_offset();

const fn make_length_minus_offset() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut i = 0usize;
    while i < 256 {
        let tag = i as u8;
        let tag_type = tag & 3;
        if tag_type == 0 {
            // Literal
            let lit_len = (tag >> 2) + 1;
            if lit_len <= 16 {
                // Common short literal: entry = len - 256 (negative)
                table[i] = (lit_len as i16) - 256;
            } else {
                // Long/extended literal: exit to slow path
                table[i] = 0x0180u16 as i16;
            }
        } else if tag_type == 1 {
            // Copy-1: len 4-11, offset has high bits in tag
            let len = (4 + ((tag >> 2) & 7)) as i16;
            let offset_high = ((tag >> 5) & 7) as i16;
            table[i] = len - (offset_high << 8);
        } else if tag_type == 2 {
            // Copy-2
            let len = (1 + (tag >> 2)) as i16;
            if len <= 16 {
                table[i] = len;
            } else {
                // Long copy-2: exit to slow path
                table[i] = 0x0180u16 as i16;
            }
        } else {
            // Copy-4: exit to slow path
            table[i] = 0x0180u16 as i16;
        }
        i += 1;
    }
    table
}

/// Extract copy offset from trailing bytes using a packed 64-bit mask constant.
/// Returns 0 for literals, low byte for copy-1, full u16 for copy-2, 0 for copy-4.
#[inline(always)]
fn extract_offset(val: u32, tag_type: usize) -> u32 {
    const MASKS: u64 = 0x0000FFFF00FF0000u64;
    let mask = ((MASKS >> (tag_type * 16)) & 0xFFFF) as u32;
    val & mask
}


/// Returns the decompressed size (in bytes) of the compressed bytes given.
///
/// `input` must be a sequence of bytes returned by a conforming Snappy
/// compressor.
///
/// # Errors
///
/// This function returns an error in the following circumstances:
///
/// * An invalid Snappy header was seen.
/// * The total space required for decompression exceeds `2^32 - 1`.
pub fn decompress_len(input: &[u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }
    Ok(Header::read(input)?.decompress_len)
}

/// Decoder is a raw decoder for decompressing bytes in the Snappy format.
///
/// This decoder does not use the Snappy frame format and simply decompresses
/// the given bytes as if it were returned from `Encoder`.
///
/// Unless you explicitly need the low-level control, you should use
/// [`read::FrameDecoder`](../read/struct.FrameDecoder.html)
/// instead, which decompresses the Snappy frame format.
#[derive(Clone, Debug, Default)]
pub struct Decoder {
    // Place holder for potential future fields.
    _dummy: (),
}

impl Decoder {
    /// Return a new decoder that can be used for decompressing bytes.
    pub fn new() -> Decoder {
        Decoder { _dummy: () }
    }

    /// Decompresses all bytes in `input` into `output`.
    ///
    /// `input` must be a sequence of bytes returned by a conforming Snappy
    /// compressor.
    ///
    /// The size of `output` must be large enough to hold all decompressed
    /// bytes from the `input`. The size required can be queried with the
    /// `decompress_len` function.
    ///
    /// On success, this returns the number of bytes written to `output`.
    ///
    /// # Errors
    ///
    /// This method returns an error in the following circumstances:
    ///
    /// * Invalid compressed Snappy data was seen.
    /// * The total space required for decompression exceeds `2^32 - 1`.
    /// * `output` has length less than `decompress_len(input)`.
    pub fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize> {
        if input.is_empty() {
            return Err(Error::Empty);
        }
        let hdr = Header::read(input)?;
        if hdr.decompress_len > output.len() {
            return Err(Error::BufferTooSmall {
                given: output.len() as u64,
                min: hdr.decompress_len as u64,
            });
        }
        let dst = &mut output[..hdr.decompress_len];
        let mut dec =
            Decompress { src: &input[hdr.len..], s: 0, dst: dst, d: 0 };
        dec.decompress()?;
        Ok(dec.dst.len())
    }

    /// Decompresses all bytes in `input` into a freshly allocated `Vec`.
    ///
    /// This is just like the `decompress` method, except it allocates a `Vec`
    /// with the right size for you. (This is intended to be a convenience
    /// method.)
    ///
    /// This method returns an error under the same circumstances that
    /// `decompress` does.
    pub fn decompress_vec(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let mut buf = vec![0; decompress_len(input)?];
        let n = self.decompress(input, &mut buf)?;
        buf.truncate(n);
        Ok(buf)
    }
}

/// Decompress is the state of the Snappy compressor.
struct Decompress<'s, 'd> {
    /// The original compressed bytes not including the header.
    src: &'s [u8],
    /// The current position in the compressed bytes.
    s: usize,
    /// The output buffer to write the decompressed bytes.
    dst: &'d mut [u8],
    /// The current position in the decompressed buffer.
    d: usize,
}

impl<'s, 'd> Decompress<'s, 'd> {
    /// Decompresses snappy compressed bytes in `src` to `dst`.
    ///
    /// This assumes that the header has already been read and that `dst` is
    /// big enough to store all decompressed bytes.
    fn decompress(&mut self) -> Result<()> {
        while self.s < self.src.len() {
            let byte = self.src[self.s];
            self.s += 1;
            if byte & 0b000000_11 == 0 {
                let len = (byte >> 2) as usize + 1;
                self.read_literal(len)?;
            } else {
                self.read_copy(byte)?;
            }
        }
        if self.d != self.dst.len() {
            return Err(Error::HeaderMismatch {
                expected_len: self.dst.len() as u64,
                got_len: self.d as u64,
            });
        }
        Ok(())
    }

    /// Decompresses a literal from `src` starting at `s` to `dst` starting at
    /// `d` and returns the updated values of `s` and `d`. `s` should point to
    /// the byte immediately proceding the literal tag byte.
    ///
    /// `len` is the length of the literal if it's <=60. Otherwise, it's the
    /// length tag, indicating the number of bytes needed to read a little
    /// endian integer at `src[s..]`. i.e., `61 => 1 byte`, `62 => 2 bytes`,
    /// `63 => 3 bytes` and `64 => 4 bytes`.
    ///
    /// `len` must be <=64.
    #[inline(always)]
    fn read_literal(&mut self, len: usize) -> Result<()> {
        debug_assert!(len <= 64);
        let mut len = len as u64;
        // As an optimization for the common case, if the literal length is
        // <=16 and we have enough room in both `src` and `dst`, copy the
        // literal using unaligned loads and stores.
        //
        // We pick 16 bytes with the hope that it optimizes down to a 128 bit
        // load/store.
        if len <= 16
            && self.s + 16 <= self.src.len()
            && self.d + 16 <= self.dst.len()
        {
            unsafe {
                // SAFETY: We know both src and dst have at least 16 bytes of
                // wiggle room after s/d, even if `len` is <16, so the copy is
                // safe.
                let srcp = self.src.as_ptr().add(self.s);
                let dstp = self.dst.as_mut_ptr().add(self.d);
                // Hopefully uses SIMD registers for 128 bit load/store.
                ptr::copy_nonoverlapping(srcp, dstp, 16);
            }
            self.d += len as usize;
            self.s += len as usize;
            return Ok(());
        }
        // When the length is bigger than 60, it indicates that we need to read
        // an additional 1-4 bytes to get the real length of the literal.
        if len >= 61 {
            // If there aren't at least 4 bytes left to read then we know this
            // is corrupt because the literal must have length >=61.
            if self.s as u64 + 4 > self.src.len() as u64 {
                return Err(Error::Literal {
                    len: 4,
                    src_len: (self.src.len() - self.s) as u64,
                    dst_len: (self.dst.len() - self.d) as u64,
                });
            }
            // Since we know there are 4 bytes left to read, read a 32 bit LE
            // integer and mask away the bits we don't need.
            let byte_count = len as usize - 60;
            let mask = u32::MAX >> ((4 - byte_count as u32) << 3);
            len = bytes::read_u32_le(&self.src[self.s..]) as u64;
            len = (len & mask as u64) + 1;
            self.s += byte_count;
        }
        // If there's not enough buffer left to load or store this literal,
        // then the input is corrupt.
        // if self.s + len > self.src.len() || self.d + len > self.dst.len() {
        if ((self.src.len() - self.s) as u64) < len
            || ((self.dst.len() - self.d) as u64) < len
        {
            return Err(Error::Literal {
                len: len,
                src_len: (self.src.len() - self.s) as u64,
                dst_len: (self.dst.len() - self.d) as u64,
            });
        }
        unsafe {
            // SAFETY: We've already checked the bounds, so we know this copy
            // is correct.
            let srcp = self.src.as_ptr().add(self.s);
            let dstp = self.dst.as_mut_ptr().add(self.d);
            ptr::copy_nonoverlapping(srcp, dstp, len as usize);
        }
        self.s += len as usize;
        self.d += len as usize;
        Ok(())
    }

    /// Reads a copy from `src` and writes the decompressed bytes to `dst`. `s`
    /// should point to the byte immediately proceding the copy tag byte.
    #[inline(always)]
    fn read_copy(&mut self, tag_byte: u8) -> Result<()> {
        // Find the copy offset and len, then advance the input past the copy.
        // The rest of this function deals with reading/writing to output only.
        let entry = TAG_LOOKUP_TABLE.entry(tag_byte);
        let offset = entry.offset(self.src, self.s)?;
        let len = entry.len();
        self.s += entry.num_tag_bytes();

        // What we really care about here is whether `d == 0` or `d < offset`.
        // To save an extra branch, use `d < offset - 1` instead. If `d` is
        // `0`, then `offset.wrapping_sub(1)` will be usize::MAX which is also
        // the max value of `d`.
        if self.d <= offset.wrapping_sub(1) {
            return Err(Error::Offset {
                offset: offset as u64,
                dst_pos: self.d as u64,
            });
        }
        // When all is said and done, dst is advanced to end.
        let end = self.d + len;
        // When the copy is small and the offset is at least 8 bytes away from
        // `d`, then we can decompress the copy with two 64 bit unaligned
        // loads/stores.
        if offset >= 8 && len <= 16 && self.d + 16 <= self.dst.len() {
            unsafe {
                let dstp = self.dst.as_mut_ptr().add(self.d);
                let srcp = dstp.sub(offset);
                ptr::copy_nonoverlapping(srcp, dstp, 8);
                ptr::copy_nonoverlapping(srcp.add(8), dstp.add(8), 8);
            }
        } else if end + 24 <= self.dst.len() {
            unsafe {
                let dest_len = self.dst.len();
                let mut dstp = self.dst.as_mut_ptr().add(self.d);
                let mut srcp = dstp.sub(offset);
                loop {
                    debug_assert!(dstp >= srcp);
                    let diff = (dstp as usize) - (srcp as usize);
                    if diff >= 16 {
                        break;
                    }
                    debug_assert!(self.d + 16 <= dest_len);
                    ptr::copy(srcp, dstp, 16);
                    self.d += diff as usize;
                    dstp = dstp.add(diff);
                }
                while self.d < end {
                    ptr::copy_nonoverlapping(srcp, dstp, 16);
                    srcp = srcp.add(16);
                    dstp = dstp.add(16);
                    self.d += 16;
                }
            }
        } else {
            if end > self.dst.len() {
                return Err(Error::CopyWrite {
                    len: len as u64,
                    dst_len: (self.dst.len() - self.d) as u64,
                });
            }
            while self.d != end {
                self.dst[self.d] = self.dst[self.d - offset];
                self.d += 1;
            }
        }
        self.d = end;
        Ok(())
    }
}

/// Header represents the single varint that starts every Snappy compressed
/// block.
#[derive(Debug)]
struct Header {
    /// The length of the header in bytes (i.e., the varint).
    len: usize,
    /// The length of the original decompressed input in bytes.
    decompress_len: usize,
}

impl Header {
    /// Reads the varint header from the given input.
    ///
    /// If there was a problem reading the header then an error is returned.
    /// If a header is returned then it is guaranteed to be valid.
    #[inline(always)]
    fn read(input: &[u8]) -> Result<Header> {
        let (decompress_len, header_len) = bytes::read_varu64(input);
        if header_len == 0 {
            return Err(Error::Header);
        }
        if decompress_len > MAX_INPUT_SIZE {
            return Err(Error::TooBig {
                given: decompress_len as u64,
                max: MAX_INPUT_SIZE,
            });
        }
        Ok(Header { len: header_len, decompress_len: decompress_len as usize })
    }
}

/// A lookup table for quickly computing the various attributes derived from
/// a tag byte. The attributes are most useful for the three "copy" tags
/// and include the length of the copy, part of the offset (for copy 1-byte
/// only) and the total number of bytes proceding the tag byte that encode
/// the other part of the offset (1 for copy 1, 2 for copy 2 and 4 for copy 4).
///
/// More specifically, the keys of the table are u8s and the values are u16s.
/// The bits of the values are laid out as follows:
///
/// xxaa abbb xxcc cccc
///
/// Where `a` is the number of bytes, `b` are the three bits of the offset
/// for copy 1 (the other 8 bits are in the byte proceding the tag byte; for
/// copy 2 and copy 4, `b = 0`), and `c` is the length of the copy (max of 64).
///
/// We could pack this in fewer bits, but the position of the three `b` bits
/// lines up with the most significant three bits in the total offset for copy
/// 1, which avoids an extra shift instruction.
///
/// In sum, this table is useful because it reduces branches and various
/// arithmetic operations.
struct TagLookupTable([u16; 256]);

impl TagLookupTable {
    /// Look up the tag entry given the tag `byte`.
    #[inline(always)]
    fn entry(&self, byte: u8) -> TagEntry {
        TagEntry(self.0[byte as usize] as usize)
    }
}

/// Represents a single entry in the tag lookup table.
///
/// See the documentation in `TagLookupTable` for the bit layout.
///
/// The type is a `usize` for convenience.
struct TagEntry(usize);

impl TagEntry {
    /// Return the total number of bytes proceding this tag byte required to
    /// encode the offset.
    fn num_tag_bytes(&self) -> usize {
        self.0 >> 11
    }

    /// Return the total copy length, capped at 255.
    fn len(&self) -> usize {
        self.0 & 0xFF
    }

    /// Return the copy offset corresponding to this copy operation. `s` should
    /// point to the position just after the tag byte that this entry was read
    /// from.
    ///
    /// This requires reading from the compressed input since the offset is
    /// encoded in bytes proceding the tag byte.
    fn offset(&self, src: &[u8], s: usize) -> Result<usize> {
        let num_tag_bytes = self.num_tag_bytes();
        let trailer =
            // It is critical for this case to come first, since it is the
            // fast path. We really hope that this case gets branch
            // predicted.
            if s + 4 <= src.len() {
                unsafe {
                    // SAFETY: The conditional above guarantees that
                    // src[s..s+4] is valid to read from.
                    let p = src.as_ptr().add(s);
                    // Mask to extract only the bytes we need from the u32.
                    // num_tag_bytes is 1, 2, or 4 for copy tags, so the
                    // shift is always 24, 16, or 0 (all < 32).
                    // This avoids a table lookup on the critical path.
                    let mask = u32::MAX >> ((4 - num_tag_bytes as u32) << 3);
                    bytes::loadu_u32_le(p) as usize & mask as usize
                }
            } else if num_tag_bytes == 1 {
                if s >= src.len() {
                    return Err(Error::CopyRead {
                        len: 1,
                        src_len: (src.len() - s) as u64,
                    });
                }
                src[s] as usize
            } else if num_tag_bytes == 2 {
                if s + 1 >= src.len() {
                    return Err(Error::CopyRead {
                        len: 2,
                        src_len: (src.len() - s) as u64,
                    });
                }
                bytes::read_u16_le(&src[s..]) as usize
            } else {
                return Err(Error::CopyRead {
                    len: num_tag_bytes as u64,
                    src_len: (src.len() - s) as u64,
                });
            };
        Ok((self.0 & 0b0000_0111_0000_0000) | trailer)
    }
}
