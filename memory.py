import random
import numpy as np
import sys
import os
import multiprocessing as mp

def concurrent_rle_compression(memory_to_compress, memory_to_save, terminate):
    while terminate.qsize() == 0:
        memory = memory_to_compress.get()
        state0 = memory[0] #at index 0 is state0
        state1 = memory[3] #at index 3 is state1
        compressed_state0 = []
        for i in range(4):
            compressed_state0.append(rle_compress(state0[i]))

        compressed_state1 = []
        for i in range(4):
            compressed_state1.append(rle_compress(state1[i]))
        
        compressed_memory = (compressed_state0, memory[1], memory[2], compressed_state1, memory[4])
        memory_to_save.put(compressed_memory)

def rle_compress(frame):
    compressed_frame = []
    for y in range(84):
        count = 0
        current_number = 0
        for x in range(84):
            if x == 0:
                current_number = frame[y][x]
            if current_number == frame[y][x]:
                count += 1
            else:
                compressed_frame.append(( np.uint8(current_number), np.uint8(count) ))
                count = 1
                current_number = frame[y][x]
        compressed_frame.append(( np.uint8(current_number), np.uint8(count) ))
    return np.asarray(compressed_frame)

def concurrent_rle_decompression(memory_to_decompress, memory_to_load, terminate):
    while terminate.qsize() == 0:
        memory = memory_to_decompress.get()
        state0 = memory['state0']
        state1 = memory['state1']
        decompressed_state0 = []
        for i in range(4):
            decompressed_state0.append(rle_decompress(state0[i]))

        decompressed_state1 = []
        for i in range(4):
            decompressed_state1.append(rle_decompress(state1[i]))

        memory['state0'] = decompressed_state0
        memory['state1'] = decompressed_state1
        memory_to_load.put(memory)

def rle_decompress(compressed_frame):
    frame = np.array(np.zeros((84, 84), dtype=np.uint8))
    y_index = 0
    x_index = 0
    count = 0
    for i in range(len(compressed_frame)):
        current_number = compressed_frame[i][0]
        for j in range(compressed_frame[i][1]):
            x_index = count + j
            frame[y_index][x_index] = np.uint8(current_number)
        count += compressed_frame[i][1]
        if(x_index == 83):
            y_index += 1
            count = 0
    return frame

class Memory:
    def __init__(self, max_size, is_multiprocessing=True):
        #self.buffer = deque(maxlen = max_size)
        self.size = max_size
        self.experience = []
        self.is_multiprocessing = is_multiprocessing

        if is_multiprocessing:
            self.first = True
            self.memory_to_compress = mp.Queue()
            self.memory_to_save = mp.Queue()
            self.memory_to_load = mp.Queue()
            self.memory_to_decompress = mp.Queue()
            self.terminate = mp.Queue()

            self.number_of_processes_for_compressing = int(mp.cpu_count() / 2)
            self.number_of_processes_for_decompressing = mp.cpu_count() - self.number_of_processes_for_compressing

            self.processes_compressing = []
            self.processes_decompressing = []
            self.process_save_to_memory = mp.Process()

            for i in range(self.number_of_processes_for_compressing):
                p = mp.Process(target=concurrent_rle_compression, args=(self.memory_to_compress, self.memory_to_save, self.terminate))
                self.processes_compressing.append(p)
                p.start()
            
            for i in range(self.number_of_processes_for_decompressing):
                p = mp.Process(target=concurrent_rle_decompression, args=(self.memory_to_decompress, self.memory_to_load, self.terminate))
                self.processes_decompressing.append(p)
                p.start()

    def __del__(self):
        self.terminate.put(True)
        for p in self.processes_compressing:
            p.terminate()
        for p in self.processes_decompressing:
            p.terminate()

    def add_with_compression(self, memory):
        self.memory_to_compress.put(memory)

    def put_compressed_memory_into_memories(self):
        for num_of_memory in range(self.memory_to_save.qsize()):
            memory = self.memory_to_save.get()
            self.add(memory[0], memory[1], memory[2], memory[3], memory[4])

    def sample_with_decompression(self, batch_size):
        return

    def add(self, state0, action, reward, state1, done):
        #self.buffer.append(experience) #(state0, action, reward, state1, done)
        if len(self.experience) >= self.size:
            self.experience.pop(0)

        self.experience.append({'state0': state0,
                                'action': action,
                                'reward': reward,
                                'state1': state1,
                                'done': done})

        if len(self.experience) % 100 == 0 and len(self.experience) != self.size:
            print("{0} of {1} samples accumulated".format(len(self.experience), self.size))

    def sample(self, batch_size):   
        if not self.is_multiprocessing:
            batch = []
            for i in range(batch_size):
                batch.append(self.experience[random.randrange(0, len(self.experience))])
            return np.asarray(batch)
        else:
            if self.first:
                self.first = False
                for i in range(batch_size):
                    self.memory_to_decompress.put(self.experience[random.randrange(0, len(self.experience))])

            for i in range(batch_size):
                self.memory_to_decompress.put(self.experience[random.randrange(0, len(self.experience))])
            decompressed_batch = []
            for i in range(batch_size):
                decompressed_batch.append(self.memory_to_load.get())
            return np.asarray(decompressed_batch)

    def length(self):
       return self.experience.__len__()

    def size_in_megabytes(self):
        return sys.getsizeof(self.experience) / 1024.0