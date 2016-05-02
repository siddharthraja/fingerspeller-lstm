from os import listdir
from random import shuffle

if __name__ == '__main__':

    root_data_dir = 'hand_images/size-normalized/'
    
    letters = listdir(root_data_dir)
    desired_sequence_count = 50
    desired_test_sequence_count = 50
    seq_length = 5

    train_out_file = open('train.txt', 'a')
    test_out_file  = open('test.txt', 'a')

    for i, letter in enumerate(letters):

        for _ in range(desired_sequence_count):
            candidates = listdir(root_data_dir+letter)
            shuffle(candidates)
            candidates = candidates[:seq_length]
            candidates = ' '.join([root_data_dir + letter +'/'+  c for c in candidates])
            train_out_file.write('%s %s\n'%(candidates, str(i+1)))

        for _ in range(desired_sequence_count):
            candidates = listdir(root_data_dir+letter)
            shuffle(candidates)
            candidates = candidates[:seq_length]
            candidates = ' '.join([root_data_dir + letter +'/'+  c for c in candidates])
            test_out_file.write('%s %s\n'%(candidates, str(i+1)))
