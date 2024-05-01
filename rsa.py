#Reference: William Stallings, "Cryptography and Network Security: Principles And Practice: Chapter 9: Seventh Edition”, Pearson 2017

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class RSA:

    def __init__(self):

        (self.p, self.q),self.phi, \
            self.pr, self.pu = self.generate_keys()

        (self.expA, self.expB), \
            (self.Xp, self.Xq) = self.precompute()

        print("Initialization Complete")


    # Generates (p, q), (n, phi), (e, d)
    def generate_keys(self):

        # 1. Generates p, q & n
        l = 1 << 15
        h = 1 << 16

        lst = []
        while len(lst) < 2:
            # randomly select a large-number
            x = int(np.random.randint(low= l, high= h))

            # check
            if self.miller_rabin(x):
                lst.append(x)

        p, q = lst[0], lst[1]

        n = p * q
        phi = (p-1) * (q-1)

        # 2. Generates e & d
        e = (1 << 16) + 1 # because only 2 multiplications
        d = self.multiplicative_inverse(e, phi)

        pu = (e, n)
        pr = (d, n)

        return (p, q), phi, pr, pu


    # Performs encryption via efficient-exponentiation
    # Reference: Section 9.2 : Computational Aspects : EFFICIENT OPERATION USING THE PUBLIC KEY
    def encrypt(self, m, pu):

        # decmpose "e" into a sum of powers of 2
        p2 = self.decompose(pu[0])
        # a = the highest-power of 2 < m; it is already stored in the list "p2"
        a = int(p2[-1])

        # calculates {m^(2^i) % n} \forall i \in [1, a]
        # e.g. m, m^2, m^4, m^8, m^16, ..
        powers = [m]
        for i in range(a):
            m = m * m % pu[1]
            powers.append(m)

        # efficiently performs m^e % n
        c = 1
        for i in p2:
            c = c * powers[i] % pu[1]

        return c

    def decrypt(self, c):

        Vp = self.exponent(c, self.expA, self.p)
        Vq = self.exponent(c, self.expB, self.q)


        m = (Vp * self.Xp + Vq * self.Xq) % self.pr[1] # pr = (d, n)

        return m

    # Returns a^-1 % b, by using the Extended Euclidean Algorithm
    # Source: https://brilliant.org/wiki/extended-euclidean-algorithm/
    def multiplicative_inverse(self, a, b):

        x, y, u, v = 0, 1, 1, 0
        p = b
        while a != 0:
            q, r = b//a, b%a
            m, n = x-u*q, y-v*q
            b,a, x,y, u,v = a,r, u,v, m,n

        return x % p

    # This function is used to perform Miller-Rabin's test
    # Reference: Chapter 2, Section 2.6 TESTING FOR PRIMALITY
    def primality_test(self, a, k, q, n):

        # a = a^q % n
        a = pow(a, q, n)

        if a == 1 or a == n-1:
            return False

        n_1 = n-1
        for j in range(1, k):
            # a = a^2 % n
            a = pow(a, 2, n)

            if a == n_1:
                return False

        return True

    # Checks whether a number is prime, by using Miller-Rabin's test
    # n = input, t affects "confidence"
    # Reference: Chapter 2, Section 2.6 TESTING FOR PRIMALITY
    def miller_rabin(self, n, t= 10):

        if n == 2 or n == 3:
            return True
        elif n % 2 == 0:
            return False

        # n-1 = 2^k * q
        k = 0
        q = n -1
        while q % 2 == 0:
            q = q >> 1
            k += 1

        # select t numbers
        lst = np.random.randint(low= 2, high= n-1, size= t)
        lst = list(map(lambda x: int(x), lst))
        for a in lst:
            if self.primality_test(a, k, q, n):
                return False

        return True

    # To performs efficient-decryption, I precompute Xp, Xq, d % (p-1), d % (q-1)
    # Reference: Section 9.2 : Computational Aspects : EFFICIENT OPERATION USING THE PRIVATE KEY, p. 301
    def precompute(self):

        # d % (p-1)
        a = self.pr[0] % (self.p-1)
        # d % (q-1)
        b = self.pr[0] % (self.q-1)

        iq = self.multiplicative_inverse(self.q, self.p)
        ip = self.multiplicative_inverse(self.p, self.q)

        Xp = self.q * iq
        Xq = self.p * ip

        expA = self.decompose(a)
        expB = self.decompose(b)

        return (expA, expB), (Xp, Xq)

    # This function decomposes an integer n into a integer-sum of powers-of-2
    def decompose(self, n):

        powers = []
        exp = 0

        while n:

            # If the Lsb is set
            if n & 1:
                powers.append( exp )

            # Right-shift, to check the next bit
            n >>= 1
            exp += 1

        return powers

    """
    1. This function performs fast-exponentiation
    2. It returns v = c ** D % p
    # Reference: Section 9.2 : Computational Aspects : EXPONENTIATION IN MODULAR ARITHMETIC
    """
    def exponent(self, c, expD, p):

        # calculates c^(2^i) % n forall i \in [1, b]
        powers = [c]
        for i in range(expD[-1]):
            c = c * c % p
            powers.append(c)

        r = 1
        for i in expD:
            r = r * powers[i] % p

        return r


    #This function modifies the dimensions of the image, to enable Diffusion
    def preprocess(self, img, block= 32):

        # Save original-shape; This is used on the decrypted-image
        self.org_shape = img.shape

        # Determine whether modification is necessary
        r, c, ch = img.shape

        flag1 = (r % block != 0)
        flag2 = (c % block != 0)

        if flag1 or flag2:
            print("Modifying the image's dimensions, to apply 'diffusion'")

        if flag1:
            r += block - (r % block)
        if flag2:
            c += block - (c % block)

        new_shape = (r, c, 3)
        # This new-array will store the image, as a sub-matrix
        new = np.zeros(new_shape, dtype= np.int64)

        # copy original-image into the new-image
        r, c, ch = img.shape
        new[:r, :c, :] = img

        return new

    # This function is used to restore decrypted-image to its original dimensions
    def postprocess(self, decipher):

        r, c, _ = self.org_shape

        img = decipher[:r, :c, :]

        return img.astype(np.uint8)

    """
    This functions performs 2 types of diffusion on the input-image, as follows
    1. The image is split into submatrices of size 32x32, each is transposed
    2. The current-row is XORed with its previous-row
    """

    def diffusion(self, img, block= 32):

        (r, c, ch) = img.shape

        tmp1 = np.empty(img.shape, dtype= np.int64)
        tmp2 = np.empty(img.shape, dtype= np.int64)

        # Transposition on sub-matrices
        for i in range(0, r, block):
            for j in range(0, c, block):
                tmp1[i:i+block, j:j+block, :] = np.transpose(img[i:i+block, j:j+block, :], axes= (1,0,2))

        # XOR current-row with the previous-row
        tmp2[0, :, :] = tmp1[0, :, :]
        for i in range(1, r):
            tmp2[i, :, :] = tmp1[i, :, :] ^ tmp2[i-1, :, :]
        
        # XOR current-column with the previous-column
        tmp2 = np.transpose(tmp2, axes =(1, 0, 2))
        for i in range(1, c):
            tmp2[i, :, :] = tmp2[i, :, :] ^ tmp2[i-1, :, :]
        tmp2 = np.transpose(tmp2, axes =(1, 0, 2))
            
        # Displays the results
        fig, axis = plt.subplots(nrows= 1, ncols= 2, figsize= (15, 5))
        axis[0].imshow(tmp1)
        axis[1].imshow(tmp2)
        axis[0].set_title("After Permutation")
        axis[1].set_title("After Diffusion")
        fig.suptitle("Image before Encryption")

        return tmp2

    # This function is used to undo the effect of "diffusion" on the decrypted-image
    def coalescence(self, img, block= 32):

        r, c, ch = img.shape
        
        # XOR current-column with the previous column
        img = np.transpose(img, axes =(1, 0, 2))
        for i in reversed(range(1, c)):
            img[i, :, :] = img[i, :, :] ^ img[i-1, :, :]
        img = np.transpose(img, axes =(1, 0, 2))
            
        # XOR current-row with the previous row
        for i in reversed(range(1, r)):
            img[i, :, :] = img[i, :, :] ^ img[i-1, :, :]

        tmp = np.empty(img.shape, dtype= np.int64)

        # Transpositions sub-matrices
        for i in range(0, r, block):
            for j in range(0, c, block):
                tmp[i:i+block, j:j+block, :] = np.transpose(img[i:i+block, j:j+block, :], axes= (1,0,2))

        return tmp

    # This function is used to encrypt the image
    # The arguments are: a numPy matrix "img", the public-key "PU"
    def encrypt_image(self, img, PU):

        # "block" = size of block for 'diffusion'
        block = 32

        img = self.preprocess(img, block)
        
        img = self.diffusion( img, block)

        shape = img.shape

        # flatten image
        msg = np.reshape(img, (-1, 3))

        # encrypt
        cipher = np.empty(msg.shape, dtype= np.int64)
        
        for i in tqdm(range(len(cipher))):
            cipher[i, 0] = self.encrypt(msg[i, 0], PU)
            cipher[i, 1] = self.encrypt(msg[i, 1], PU)
            cipher[i, 2] = self.encrypt(msg[i, 2], PU)

        # restore shape
        cipher = np.reshape(cipher, shape)

        return (cipher, self.org_shape)

    """
    This function is used to decrypt the image
    The arguments are: a numPy matrix 'img' (this is the cipher),
    & the original-image's dimensions
    - The latter argument is necessary because
    1. the image's dimensions were modified (to enable 'Diffusion')
    2. the recepient of the cipher does not know the original-dimensions
    """
    def decrypt_image(self, img, org_shape):

        self.org_shape = org_shape

        shape = img.shape

        # flatten image
        cipher = np.reshape(img, (-1, 3))

        # decipher
        decipher = np.empty(cipher.shape, dtype= np.int64)
        for i in tqdm(range(len(cipher))):
            decipher[i,0] = self.decrypt(cipher[i,0])
            decipher[i,1] = self.decrypt(cipher[i,1])
            decipher[i,2] = self.decrypt(cipher[i,2])

        # restore
        decipher =  np.reshape(decipher, shape)

        decipher = self.coalescence(decipher)

        return self.postprocess(decipher)
