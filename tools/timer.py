from time import process_time_ns, perf_counter_ns
import math


class Timer:
    def __init__(self) -> None:
        self.process_time = [] # w/o sleep
        self.perf_time = [] # real time
    
    def reset(self, tick=False):
        self.process_time = [] # w/o sleep
        self.perf_time = [] # real time
        
        if tick:
            self.tick()
    
    def tick(self):
        self.process_time.append(process_time_ns())
        self.perf_time.append(perf_counter_ns())

    def tick_and_get(self):
        self.tick()
        return self._diff(self.process_time), self._diff(self.perf_time)
    
    def _diff(self, arr, total=False):
        assert len(arr) > 1
        return arr[-1] - arr[0 if total else -2]
    
    def _average(self, arr):
        assert len(arr) > 1
        return self._diff(arr, total=True) / (len(arr) - 1)
    
    def average_process_time(self, offset=0):
        return self._average(self.process_time[offset:])
    
    def average_perf_time(self, offset=0):
        return self._average(self.perf_time[offset:])

    def averages(self, offset=0):
        return self.average_process_time(offset=offset), self.average_perf_time(offset=offset)
    
    def performances(self, offset=0, verbose = False):
        def with_unit(val):
            u = int(math.log10(val)//3)
            unit = 'num'[u] + 's' if val < 1e9 else 's'
            return f'{val/(10 ** (u*3)):.3f} {unit}'
            
        proc, perf = self.average_process_time(offset), self.average_perf_time(offset)
        repr = f'[Average] Total time = {with_unit(perf)}, Process time = {with_unit(proc)}'
        
        if verbose:
            repr += f'\n\t - Raw: total = {perf} ns, process = {proc} ns'
            repr += f'\n\t - Elasped: total = {self._diff(self.perf_time[offset:])} ns, process = {self._diff(self.process_time[offset:])} ns'

        return repr