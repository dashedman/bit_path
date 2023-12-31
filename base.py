def sha1(data):
    bytes_ = ''

    h0 = 0x67452301
    h1 = 0xEFCDAB89
    h2 = 0x98BADCFE
    h3 = 0x10325476
    h4 = 0xC3D2E1F0

    for char in data:
        bytes_ += f'{ord(char):08b}'

    bits = bytes_ + "1"
    p_bits = bits
    # pad until length equals 448 mod 512
    if len(p_bits) % 512 > 448:
        p_bits += "0" * (512 - len(p_bits) % 512)
    if len(p_bits) % 512 < 448:
        p_bits += "0" * (448 - len(p_bits) % 512)
    # append the original length
    p_bits += '{0:064b}'.format(len(bits) - 1)

    def chunks(lst: str, size: int):
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    def rol(n: int, b: int):
        return ((n << b) | (n >> (32 - b))) & 0xffffffff

    for c in chunks(p_bits, 512):
        words = chunks(c, 32)
        w = [0] * 80
        for n in range(0, 16):
            w[n] = int(words[n], 2)
        for i in range(16, 80):
            w[i] = rol((w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]), 1)

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
            elif 60 <= i <= 79:
                f = b ^ c ^ d
                k = 0xCA62C1D6

            temp = rol(a, 5) + f + e + k + w[i] & 0xffffffff
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
    print(sha1("hello world"))
