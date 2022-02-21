#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiprocessing
5/18/2021
djkim
"""

import multiprocessing as mp
import time
import os
import pandas as pd
import numpy as np

# Example 1
start = time.perf_counter()

def f(sec):
    print(f'process {os.getpid()}: sleeping {sec} seconds')
    time.sleep(sec)
    print(f'done sleeping {sec} seconds')
    
num_process = 20
secs = [2 for i in range(num_process)]
pool = mp.Pool(processes=num_process)
#pool = mp.Pool(mp.cpu_count())
result = pool.map(f, secs)
    
finish = time.perf_counter()
print(f'Finished in {finish-start:.2f} seconds')
#Finished in 2.17 seconds


# Example 2
num_process = int(mp.cpu_count())

def looping(n):
    start = time.time()
    time.sleep(n)
    df = pd.DataFrame(np.random.randn(n,3), index=pd.date_range('1/1/2019',periods=n), 
                  columns=list('abc'))
    total = time.time() - start
    print('step {} = {:7.5f} seconds'.format(n, total))
    return df

pool = mp.Pool(num_process)
result = pool.map(looping, range(num_process))
pool.close()
pool.join()

summary = pd.DataFrame()
for i in range(len(result)):
    summary = pd.concat([summary, result[i]], sort=False)

summary.to_csv('SUMMARY.csv', index=False)

    