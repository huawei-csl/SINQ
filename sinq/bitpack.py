# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
# modified by SINQ authors 2025
#####################################################

import torch
from torch import uint8, int32, Tensor
import numpy as np


# Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
    # 8-bit
    ################################################
    @staticmethod
    def pack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q.to(uint8)

    @staticmethod
    def unpack_8bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        return W_q.to(dtype)

    # 4-bit
    ################################################
    @staticmethod
    def pack_4bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/2
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 2)

        return (W_q[:_step] << 4) | W_q[_step:]

    @staticmethod
    def unpack_4bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:  # uint8/2 > uint8
        _step = W_q.shape[0]
        tmp = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        # Extract 4-bit groups using modular arithmetic
        for i in range(2):
            divisor = 16 if i == 0 else 1
            tmp[i * _step : (i + 1) * _step] = ((W_q.to(torch.float16) // divisor) % 16).to(torch.uint8)

        return tmp


    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 4)

        return (
            W_q[:_step] << 6
            | W_q[_step : 2 * _step] << 4
            | W_q[2 * _step : 3 * _step] << 2
            | W_q[3 * _step :]
        )

    @staticmethod
    def unpack_2bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([4 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        # Extract 2-bit groups using modular arithmetic
        for i in range(4):
            divisor = 2**(6 - 2*i) if i < 3 else 1
            tmp[i * _step : (i + 1) * _step] = (W_q // divisor) % 4

        return tmp


    # 3-bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in: Tensor) -> Tensor:
        W_q = torch.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=int32,
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)

        W_q = (
            (W_q[:_step] << 27)
            | (W_q[1 * _step : 2 * _step] << 24)
            | (W_q[2 * _step : 3 * _step] << 21)
            | (W_q[3 * _step : 4 * _step] << 18)
            | (W_q[4 * _step : 5 * _step] << 15)
            | (W_q[5 * _step : 6 * _step] << 12)
            | (W_q[6 * _step : 7 * _step] << 9)
            | (W_q[7 * _step : 8 * _step] << 6)
            | (W_q[8 * _step : 9 * _step] << 3)
            | (W_q[9 * _step : 10 * _step])
        )

        return W_q

    @staticmethod
    def unpack_3bit_32(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        # Extract 3-bit groups using modular arithmetic
        for i in range(10):
            divisor = 2**(27 - 3*i) if i < 9 else 1
            tmp[i * _step : (i + 1) * _step] = (W_q // divisor) % 8

        return tmp

    @staticmethod
    def pack_6bit_32(W_q_in: Tensor) -> Tensor:
        """
        Tensor [N, C]  ->  Tensor [ceil(N/5), C] (dtype = int32)
        5 consecutive elements (LSB aligned) are compressed into one 32-bit word.
        """
        # ---------- 1. extend the tensor to a multiple of 5 ----------
        STEP = 5                         # 5 values → 1 word
        padded_len = int(STEP * np.ceil(W_q_in.shape[0] / STEP))
        W_q = torch.zeros(
            [padded_len, W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=torch.int32,
        )
        W_q[: len(W_q_in)] = W_q_in.long() & 0x3F     # keep 6-bit range

        # ---------- 2. shift-compress each stride of 5 values ----------
        _step = int(len(W_q) / STEP)
        packed = (
              (W_q[0 * _step : 1 * _step] << 24)
            | (W_q[1 * _step : 2 * _step] << 18)
            | (W_q[2 * _step : 3 * _step] << 12)
            | (W_q[3 * _step : 4 * _step] <<  6)
            |  W_q[4 * _step : 5 * _step]
        )
        return packed  # shape [padded_len/5 , C]

    @staticmethod
    def unpack_6bit_32(W_packed: Tensor, dtype=torch.uint8) -> Tensor:
        """
        Reverse of pack_6bit_32.
        Tensor [M, C]  ->  Tensor [5*M, C]   (dtype = uint8)
        """
        _step = W_packed.shape[0]
        unpacked = torch.empty(
            [5 * _step, W_packed.shape[1]],
            dtype=dtype,
            device=W_packed.device,
        )
        unpacked[0 * _step : 1 * _step] = (W_packed >> 24) & 0x3F
        unpacked[1 * _step : 2 * _step] = (W_packed >> 18) & 0x3F
        unpacked[2 * _step : 3 * _step] = (W_packed >> 12) & 0x3F
        unpacked[3 * _step : 4 * _step] = (W_packed >>  6) & 0x3F
        unpacked[4 * _step : 5 * _step] =  W_packed        & 0x3F
        return unpacked


    @staticmethod
    def pack_5bit_32(W_q_in: Tensor) -> Tensor:
        """pack 6 consecutive 5-bit integers into one 32-bit word"""
        STEP = 6                         # 6 values → 1 word
        padded_len = int(STEP * np.ceil(W_q_in.shape[0] / STEP))
        W_q = torch.zeros(
            [padded_len, W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=torch.int32,
        )
        W_q[:len(W_q_in)] = W_q_in.long() & 0x1F        # keep 5 bits
        _step = int(len(W_q) / STEP) 

        W_packed = (
            (W_q[0 * _step : 1 * _step] << 25)
            | (W_q[1 * _step : 2 * _step] << 20)
            | (W_q[2 * _step : 3 * _step] << 15)
            | (W_q[3 * _step : 4 * _step] << 10)
            | (W_q[4 * _step : 5 * _step] << 5)
            |  W_q[5 * _step : 6 * _step]
        )
        return W_packed         # [padded_len//6 , C]

    @staticmethod
    def unpack_5bit_32(W_q: Tensor, dtype=torch.uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([6 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q >> 25) & 0x1F
        tmp[1 * _step : 2 * _step] = (W_q >> 20) & 0x1F
        tmp[2 * _step : 3 * _step] = (W_q >> 15) & 0x1F
        tmp[3 * _step : 4 * _step] = (W_q >> 10) & 0x1F
        tmp[4 * _step : 5 * _step] = (W_q >> 5)  & 0x1F
        tmp[5 * _step : 6 * _step] =  W_q        & 0x1F
        return tmp

    # 1-bit
    ################################################
    @staticmethod
    def pack_1bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 8)

        return (
            W_q[:_step] << 7
            | W_q[1 * _step : 2 * _step] << 6
            | W_q[2 * _step : 3 * _step] << 5
            | W_q[3 * _step : 4 * _step] << 4
            | W_q[4 * _step : 5 * _step] << 3
            | W_q[5 * _step : 6 * _step] << 2
            | W_q[6 * _step : 7 * _step] << 1
            | W_q[7 * _step : 8 * _step]
        )

    @staticmethod
    def unpack_1bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([8 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q & 0b10000000) >> 7
        tmp[1 * _step : 2 * _step] = (W_q & 0b01000000) >> 6
        tmp[2 * _step : 3 * _step] = (W_q & 0b00100000) >> 5
        tmp[3 * _step : 4 * _step] = (W_q & 0b00010000) >> 4
        tmp[4 * _step : 5 * _step] = (W_q & 0b00001000) >> 3
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000100) >> 2
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000010) >> 1
        tmp[7 * _step : 8 * _step] = W_q & 0b00000001

        return tmp

    @staticmethod

    def pack_4bit_32_signed(W_s4: torch.Tensor) -> torch.Tensor:

        """

        Pack signed int4 weights (K, N) -> (K//8, N) INT32.

        Each int32 carries 8 consecutive 4-bit values along K.

        W_s4 must be int32 in [-8, 7]. We store two's complement nibbles (val & 0xF).

        """

        assert W_s4.dtype in (torch.int32, torch.int16, torch.int8), "expect integer tensor"

        K, N = W_s4.shape

        assert K % 8 == 0, "K must be a multiple of 8 for int4 packing"

        W = W_s4.to(torch.int32).view(K // 8, 8, N)  # (K//8, 8, N)



        def nib(i):  # two's complement nibble

            return (W[:, i, :] & 0xF).to(torch.int32)



        packed = (nib(0)

                  | (nib(1) << 4)

                  | (nib(2) << 8)

                  | (nib(3) << 12)

                  | (nib(4) << 16)

                  | (nib(5) << 20)

                  | (nib(6) << 24)

                  | (nib(7) << 28))

        return packed.contiguous()



    @staticmethod

    def unpack_4bit_32_signed(W_packed: torch.Tensor) -> torch.Tensor:

        """

        Reverse of pack_4bit_32_signed.

        (K//8, N) INT32 -> (K, N) INT32 with values in [-8, 7].

        """

        M, N = W_packed.shape

        out = torch.empty((M * 8, N), dtype=torch.int32, device=W_packed.device)

        for i in range(8):

            nib = (W_packed >> (4 * i)) & 0xF   # 0..15

            # convert back to signed in [-8,7]

            out[i::8] = nib.where(nib < 8, nib - 16)

        return out
