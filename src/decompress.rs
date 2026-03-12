use std::ptr;

use crate::bytes;
use crate::error::{Error, Result};
use crate::tag;
use crate::MAX_INPUT_SIZE;

/// A lookup table for quickly computing the various attributes derived from a
/// tag byte. See the comment above `TagLookupTable` for the bit layout.
const TAG_LOOKUP_TABLE: [u16; 256] = tag::TAG_LOOKUP_TABLE;

/// Copy up to 64 bytes using unrolled 16-byte copies.
/// `src` and `dst` must not overlap in each 16-byte chunk.
/// Use `wide_copy_long` when src and dst are guaranteed >= 32 apart.
#[inline(always)]
unsafe fn wide_copy(src: *const u8, dst: *mut u8, len: usize) {
    debug_assert!(len <= 64);
    ptr::copy_nonoverlapping(src, dst, 16);
    ptr::copy_nonoverlapping(src.add(16), dst.add(16), 16);
    if len > 32 {
        ptr::copy_nonoverlapping(src.add(32), dst.add(32), 16);
        if len > 48 {
            ptr::copy_nonoverlapping(src.add(48), dst.add(48), 16);
        }
    }
}

/// Copy up to 64 bytes using 32-byte copies (ldp/stp q pairs on ARM).
/// Requires src and dst to be at least 32 bytes apart (no overlap).
#[inline(always)]
unsafe fn wide_copy_long(src: *const u8, dst: *mut u8, len: usize) {
    debug_assert!(len <= 64);
    ptr::copy_nonoverlapping(src, dst, 32);
    if len > 32 {
        ptr::copy_nonoverlapping(src.add(32), dst.add(32), 32);
    }
}

/// Extract the offset mask for a given tag_type (1, 2, or 3).
/// Returns a mask: tag_type=1 → 0xFF, tag_type=2 → 0xFFFF, tag_type=3 → 0.
///
/// On ARM, uses a packed u64 constant with shift (avoids memory load).
/// On x86, uses an array lookup (avoids 10-byte movabs + variable shift).
#[inline(always)]
fn extract_offset_mask(tag_type: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        const MASKS_PACKED: u64 = 0x0000FFFF00FF0000u64;
        ((MASKS_PACKED >> (tag_type * 16)) & 0xFFFF) as u32
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        const MASKS: [u32; 4] = [0, 0xFF, 0xFFFF, 0];
        MASKS[tag_type]
    }
}

/// Dispatch a copy of `len` bytes from `dst - offset` to `dst`.
///
/// Tries fast wide-copy paths (non-overlapping 8/16/32 byte chunks).
/// Falls back to `overlapping_copy` when offset < 16.
///
/// Caller must ensure at least `len + 24` bytes of writable space at `dst`
/// and at least `offset` valid bytes preceding `dst`.
#[inline(always)]
unsafe fn copy_dispatch(dst: *mut u8, offset: usize, len: usize) {
    let srcp = dst.sub(offset);
    if len <= 16 && offset >= 8 {
        ptr::copy_nonoverlapping(srcp, dst, 8);
        ptr::copy_nonoverlapping(srcp.add(8), dst.add(8), 8);
    } else if offset >= 32 {
        wide_copy_long(srcp, dst, len);
    } else if offset >= 16 {
        wide_copy(srcp, dst, len);
    } else {
        overlapping_copy(dst, offset, len);
    }
}

/// Copy `len` bytes from `dst - offset` into `dst`, handling overlapping
/// regions by expanding with `ptr::copy` until the gap >= 16, then
/// switching to non-overlapping 16-byte chunks.
///
/// Caller must ensure `dst + len + 24` is writable and that `dst` is
/// preceded by at least `offset` valid bytes.
#[inline(always)]
unsafe fn overlapping_copy(dst: *mut u8, offset: usize, len: usize) {
    let end = dst.add(len);
    let mut dstp = dst;
    let mut srcp = dst.sub(offset);
    loop {
        let diff = (dstp as usize) - (srcp as usize);
        if diff >= 16 {
            break;
        }
        ptr::copy(srcp, dstp, 16);
        dstp = dstp.add(diff);
    }
    while dstp < end {
        ptr::copy_nonoverlapping(srcp, dstp, 16);
        srcp = srcp.add(16);
        dstp = dstp.add(16);
    }
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
        unsafe {
            self.decompress_fast()?;
        }
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

    /// Fast decompression loop using raw pointers for common cases.
    ///
    /// Fast decompression loop using raw pointers for common cases.
    ///
    /// The loop condition guarantees sufficient headroom in both source and
    /// destination buffers to eliminate most bounds checks from the loop body:
    /// - `s + 17 <= src_len`: ensures 16 bytes of literal data + 1 tag byte
    ///   can always be read, and 4 bytes of copy offset data (since 17 > 5).
    /// - `d + 88 <= dst_len`: ensures max copy (64 bytes) + overlapping_copy
    ///   wiggle room (24 bytes) always fits, so no destination checks needed.
    #[inline(always)]
    unsafe fn decompress_fast(&mut self) -> Result<()> {
        let src = self.src.as_ptr();
        let dst_base = self.dst.as_mut_ptr();
        let src_len = self.src.len();
        let dst_len = self.dst.len();

        if src_len < 17 || dst_len < 88 {
            return Ok(());
        }

        // Use raw pointers for the hot loop to avoid base+offset additions.
        let mut ip = src.add(self.s);
        let mut op = dst_base.add(self.d);
        let ip_limit = src.add(src_len - 17);
        let op_limit = dst_base.add(dst_len - 88);
        let src_end = src.add(src_len);
        let dst_base_addr = dst_base as usize;

        let mut preload = *ip as u32;

        loop {
            let byte = preload as u8;
            // Track whether we need to reload preload from memory
            // (literals always, Copy4 always, Copy1/Copy2 never).
            let mut reload = true;

            if byte & 3 != 0 {
                let entry_val = TAG_LOOKUP_TABLE[byte as usize] as usize;
                let tag_type = (byte & 3) as usize;
                let num_tag_bytes = tag_type + (tag_type == 3) as usize;
                let len = entry_val & 0xFF;
                ip = ip.add(1);

                let loaded = bytes::loadu_u32_le(ip);
                let extracted =
                    (loaded & extract_offset_mask(tag_type)) as usize;
                let offset = (entry_val & 0x700) | extracted;
                ip = ip.add(num_tag_bytes);

                // Compute copy source once; reuse for bounds check and copy.
                let srcp = op.sub(offset);
                if (srcp as usize) < dst_base_addr || offset == 0 {
                    self.s = ip.offset_from(src) as usize;
                    self.d = op.offset_from(dst_base) as usize;
                    return Err(Error::Offset {
                        offset: offset as u64,
                        dst_pos: self.d as u64,
                    });
                }

                copy_dispatch(op, offset, len);
                op = op.add(len);

                // Preload next tag from the trailer for Copy1/Copy2.
                // For Copy4 (num_tag_bytes=4), shift is 32 → result is 0,
                // but reload=true overwrites it anyway.
                preload = loaded >> (tag_type as u32 * 8);
                reload = tag_type == 3;
            } else {
                let len = (byte >> 2) as usize + 1;
                ip = ip.add(1);
                if len <= 16 {
                    ptr::copy_nonoverlapping(ip, op, 16);
                    ip = ip.add(len);
                    op = op.add(len);
                } else if len <= 60
                    && (ip as usize + len + 16) <= (src_end as usize)
                {
                    wide_copy_long(ip, op, len);
                    ip = ip.add(len);
                    op = op.add(len);
                } else {
                    self.s = ip.offset_from(src) as usize;
                    self.d = op.offset_from(dst_base) as usize;
                    self.read_literal(len)?;
                    ip = src.add(self.s);
                    op = dst_base.add(self.d);
                }
            }

            // Single unified bounds check and preload for all paths.
            if ip > ip_limit || op > op_limit {
                break;
            }
            if reload {
                preload = *ip as u32;
            }
        }
        self.s = ip.offset_from(src) as usize;
        self.d = op.offset_from(dst_base) as usize;
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
        let entry_val = TAG_LOOKUP_TABLE[tag_byte as usize] as usize;
        let tag_type = (tag_byte & 3) as usize;
        let num_tag_bytes = tag_type + (tag_type == 3) as usize;
        let len = entry_val & 0xFF;

        // Read offset from compressed input.
        let trailer = if self.s + 4 <= self.src.len() {
            unsafe {
                let p = self.src.as_ptr().add(self.s);
                let mask = u32::MAX >> ((4 - num_tag_bytes as u32) << 3);
                bytes::loadu_u32_le(p) as usize & mask as usize
            }
        } else if num_tag_bytes == 1 {
            if self.s >= self.src.len() {
                return Err(Error::CopyRead {
                    len: 1,
                    src_len: (self.src.len() - self.s) as u64,
                });
            }
            self.src[self.s] as usize
        } else if num_tag_bytes == 2 {
            if self.s + 1 >= self.src.len() {
                return Err(Error::CopyRead {
                    len: 2,
                    src_len: (self.src.len() - self.s) as u64,
                });
            }
            bytes::read_u16_le(&self.src[self.s..]) as usize
        } else {
            return Err(Error::CopyRead {
                len: num_tag_bytes as u64,
                src_len: (self.src.len() - self.s) as u64,
            });
        };
        let offset = (entry_val & 0x700) | trailer;
        self.s += num_tag_bytes;

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
        let end = self.d + len;
        if end + 24 <= self.dst.len() {
            unsafe {
                copy_dispatch(self.dst.as_mut_ptr().add(self.d), offset, len);
            }
        } else {
            if end > self.dst.len() {
                return Err(Error::CopyWrite {
                    len: len as u64,
                    dst_len: (self.dst.len() - self.d) as u64,
                });
            }
            // Byte-by-byte fallback for the last few bytes.
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
