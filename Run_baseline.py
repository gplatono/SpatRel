import sys
import os
from baseline import run
from rel_baseline import run1
import csv

run = run1
structure = [50, 50]
repetition = 2
fields = ["h1", 'h2', 'h3', 'left', 'right', 'above', 'below', 'front',
              'behind', 'over', 'under', 'in', 'touch', 'at', 'between',
              'near', 'on top of', 'beside', 'facing', 'total']

n = len(structure)
l = [7, 6]
reset_val = 5
#l=[1, 1, 1]



def call(i, outputs, write):
    while l[i] <= structure[i]:
        if i < n-1:
            call(i+1, outputs, write)
            l[i] += 1
        else:
            for j in range(repetition):
                try:
                    print("Structure: ", l, ", Repetition: ", j)
                    # sys.stdout.flush()
                    # save_stdout = sys.stdout
                    # sys.stdout = open(os.devnull, 'w')
                    rel_acc, acc = run(l)
                    # sys.stdout = save_stdout
                    #outputs += [l + rel_acc + [acc]]
                    write.writerows([l + rel_acc + [acc]])
                    print("rel_acc: ", rel_acc)
                    print("acc: ", acc)
                except Exception as exc:
                    print(exc)
                    continue

            l[i] += 1
    l[i] = reset_val

if __name__ == '__main__':

    outputs = []
    f = open('StrctureData.csv', 'a', newline='')
    write = csv.writer(f)
    write.writerow(fields)
    call(0, outputs, write)
    # print(outputs)
    # with open('StrctureData.csv', 'a', newline='') as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(outputs)