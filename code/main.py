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
    
    sort_data = [d if r == d[1] else for d in data for r in range(3) if r == d[1]]
    
    print(sort_data)


if __name__ == '__main__':
    main()
