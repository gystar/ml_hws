import math


def BLEU(y, t):
    # y:生成语句
    # t：目标语句
    # 返回BELU的值
    c = len(y)
    r = len(t)
    right_count = 0
    for ch in y:
        if ch in t:
            right_count += 1
    ret = right_count / c
    ret *= 1 if c > r else math.exp(1 - r / c)
    return ret


##test
if __name__ == "__main__":
    a = BLEU(["1", "2", "2", "4"], ["1", "2", "3"])
    print(a)
