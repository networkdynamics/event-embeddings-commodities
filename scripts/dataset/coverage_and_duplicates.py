import random

def main():
    k = 2
    sims = 10000

    nums = {
        'missing': 0,
        'covered': 0,
        'duplicates': 0
    }

    for _ in range(sims):
        slots = [0] * 100
        for _ in range(k):
            coverage = random.sample(range(100), 60)
            for i in coverage:
                slots[i] += 1

        dup = 0
        cov = 0
        mis = 0
        for slot in slots:
            if slot == 2:
                dup += 1
            if slot > 0:
                cov += 1
            if slot == 0:
                mis += 1

        nums['missing'] += mis / sims
        nums['covered'] += cov / sims
        nums['duplicates'] += dup / sims

    print(nums)

if __name__ == '__main__':
    main()