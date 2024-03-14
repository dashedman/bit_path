BitList = list[bool | float]
H0_START = 0x67452301
H1_START = 0xEFCDAB89
H2_START = 0x98BADCFE
H3_START = 0x10325476
H4_START = 0xC3D2E1F0


def char_to_binary(char: str):
    return f'{ord(char):08b}'


def str_bits_to_list(str_bits: str) -> BitList:
    return [bool(int(str_bit)) for str_bit in str_bits]


def int32_to_list(x: int) -> BitList:
    return str_bits_to_list(f'{x:032b}')


def list_to_int(bits: BitList) -> int:
    num = 0
    for bit in bits:
        if not isinstance(bit, bool):
            raise Exception('Bad bit for converting to int!')
        num = (num << 1) | bit
    return num


def chunks(lst: BitList, size: int):
    return [lst[offset:offset + size] for offset in range(0, len(lst), size)]


def sha1(data):
    bytes_: BitList = []

    h0 = H0_START
    h1 = H1_START
    h2 = H2_START
    h3 = H3_START
    h4 = H4_START

    for char in data:
        bytes_ += str_bits_to_list(char_to_binary(char))

    bits = bytes_ + [True]
    p_bits = bits.copy()
    # pad until length equals 448 mod 512
    if len(p_bits) % 512 > 448:
        p_bits += [False] * (512 - len(p_bits) % 512)
    if len(p_bits) % 512 < 448:
        p_bits += [False] * (448 - len(p_bits) % 512)
    # append the original length
    p_bits += str_bits_to_list(f'{len(bits) - 1:064b}')


    def rol(n: int, b: int):
        return ((n << b) | (n >> (32 - b))) & 0xffffffff

    def rol_list(n: BitList, b: int) -> BitList:
        return n[b:] + n[:b]

    def xor_list(lst1: BitList, lst2: BitList):
        return [b1 ^ b2 for b1, b2 in zip(lst1, lst2)]

    for c in chunks(p_bits, 512):
        words = chunks(c, 32)
        w = [int32_to_list(0)] * 80
        for n in range(0, 16):
            w[n] = words[n]
        for i in range(16, 80):
            w[i] = rol_list(
                xor_list(
                    xor_list(
                        xor_list(
                            w[i - 3],
                            w[i - 8]
                        ),
                        w[i - 14]
                    ),
                    w[i - 16]
                ), 1
            )

        a = h0
        b = h1
        c = h2
        d = h3
        e = h4

        # Main loop
        for i in range(0, 80):
            if 0 <= i <= 19:
                f = (b & c) | ((~b) & d)
                k = 0x5A827999
            elif 20 <= i <= 39:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif 40 <= i <= 59:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = b ^ c ^ d
                k = 0xCA62C1D6

            # print(i, rol(a, 5), f, e, k, list_to_int(w[i]))
            # print(
            #     i,
            #     (rol(a, 5) + f) & 0xffffffff,
            #     (((rol(a, 5) + f) & 0xffffffff) + e) & 0xffffffff,
            #     (((((rol(a, 5) + f) & 0xffffffff) + e) & 0xffffffff) + k) & 0xffffffff
            # )
            temp = (
                        (
                            (rol(a, 5) + f & 0xffffffff)
                            + e & 0xffffffff
                        ) + k & 0xffffffff
                   ) + list_to_int(w[i]) & 0xffffffff
            temp2 = rol(a, 5) + f + e + k + list_to_int(w[i]) & 0xffffffff

            print(i, temp, temp2)
            e = d
            d = c
            c = rol(b, 30)
            b = a
            a = temp

        h0 = h0 + a & 0xffffffff
        h1 = h1 + b & 0xffffffff
        h2 = h2 + c & 0xffffffff
        h3 = h3 + d & 0xffffffff
        h4 = h4 + e & 0xffffffff

    return '%08x%08x%08x%08x%08x' % (h0, h1, h2, h3, h4)


if __name__ == '__main__':
    print(sha1(""))
