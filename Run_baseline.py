import sys
import os
from baseline import run
import csv

structure = [15, 15, 15]
repetition = 2

n = len(structure)
l = [5] * n

def call(i, outputs):
    while l[i] <= structure[i]:
        if i < n-1:
            call(i+1, outputs)
            l[i] += 1
        else:
            for j in range(repetition):
                print("Structure: ", l, ", Repetition: ", j)
                # sys.stdout.flush()
                # save_stdout = sys.stdout
                # sys.stdout = open(os.devnull, 'w')
                rel_acc, acc = run(l)
                # sys.stdout = save_stdout
                outputs += [l + rel_acc + [acc]]
                print("rel_acc: ", rel_acc)
                print("acc: ", acc)

            l[i] += 1
    l[i] = 5

if __name__ == '__main__':
    fields = ["h1", 'h2', 'h3', 'left', 'right', 'above', 'below', 'front',
              'behind', 'over', 'under', 'in', 'touch', 'at', 'between',
              'near', 'on top of', 'beside', 'facing', 'total']
    outputs = []
    call(0, outputs)
    print(outputs)
    with open('StrctureData.csv', 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(outputs)