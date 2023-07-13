import numpy as np
from collections import deque
# import subprocess
# bashCommand = "!g++ -std=c++17 checker.cpp -o A.o && ./A.o < temp.txt > res.txt"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

class Maze:
    def __init__(self):
        self.nRows = 21
        self.nCols = 31 
        self.INF = float("INF")
        self.maze = np.zeros((self.nRows-2, self.nCols-2))
        self.maze = np.pad(self.maze, pad_width=1, mode='constant',
                constant_values=self.INF)

    def update(self, x, y, action):
        self.maze[1+x][1+y] = self.INF if action else 0

    def reset(self):
        self.maze = np.zeros((self.nRows-2, self.nCols-2))
        self.maze = np.pad(self.maze, pad_width=1, mode='constant',
                constant_values=self.INF)

    def get_reward(self):
        # for x in range(self.nRows-2):
            # for y in range(self.nCols-2):
                # maze[x+1][y+1] = self.INF if maze[x][y] == 1 else 0
        if self.maze[1][1] == self.INF or self.maze[-2][-2] == self.INF:
            return -100000000 
        if not self.thereIsAPath():
            return 0
        else:
            return self.count()
    #   with open(filename, "w") as txt_file:
        # for line in maze:
            # txt_file.write(''.join([('#' if c == 1 else '.') for c in line]) + "\n")

    def thereIsAPath(self):
        q = deque()
        q.append((1,1))
        self.maze[1][1] = 1
        while len(q) > 0:
            r, c = q.popleft()
            if self.maze[r+1][c] == 0:
                self.maze[r+1][c] = 1
                q.append((r+1, c))
            if self.maze[r-1][c] == 0:
                self.maze[r-1][c] = 1
                q.append((r-1, c))
            if self.maze[r][c+1] == 0:
                self.maze[r][c+1] = 1
                q.append((r, c+1))
            if self.maze[r][c-1] == 0:
                self.maze[r][c-1] = 1
                q.append((r, c-1))
        result = bool(self.maze[self.nRows-2][self.nCols-2])
        self.restore()
        return result
        
    def restore(self):
        for r in range(self.nRows):
            for c in range(self.nCols):
                self.maze[r][c] = self.INF if self.maze[r][c] == self.INF else 0
        
    def count(self):
        r, c = 1, 1
        dir = 0
        answ = 0
        dr = (1, 0, -1,  0)
        dc = (0, 1,  0, -1)
        while True:
            self.maze[r][c] += 1
            min1 = min(self.maze[r-1][c], self.maze[r+1][c])
            min2 = min(self.maze[r][c-1], self.maze[r][c+1])
            mn = min(min1,min2)
            if self.maze[r+dr[dir]][c+dc[dir]] != mn:
                for next in range(4):
                    if self.maze[r+dr[next]][c+dc[next]] == mn:
                        dir = next
                        break
            answ += 1
            r += dr[dir]
            c += dc[dir]
            if r == self.nRows-2 and c == self.nCols-2:
                break
        self.restore()
        return answ