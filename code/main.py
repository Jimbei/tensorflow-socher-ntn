import random
import string


def main():
    data = []
    for i in range(10):
        e1 = random.choice(string.ascii_lowercase)
        e2 = random.choice(string.ascii_lowercase)
        r = random.randint(0, 2)
        data.append([e1, r, e2])
    
    print(data)
    
    sort_data = [[T for T in data if r == T[1]] for r in range(3)]
    for i in sort_data:
        print(i)


if __name__ == '__main__':
    main()
