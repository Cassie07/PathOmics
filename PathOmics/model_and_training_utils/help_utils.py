
def print_result_as_table(result_dict, num_fold):
    
    print('+-------+-------------+--------------+')
    print('|  Fold | Best epoch  | Best c-index |')
    print('+-------+-------------+--------------+')
    for i in range(num_fold):
        print('|   {}   |      {}      |    {:.4f}    |'.format(i + 1, result_dict[i][0], result_dict[i][1]))
        print('+-------+-------------+--------------+')
        
        
        
def print_result_as_table_rand_seed(result_dict, rand_seed_list):
    
    print('+--------+-------------+--------------+')
    print('|  Seed  | Best epoch  | Best c-index |')
    print('+--------+-------------+--------------+')
    for i in rand_seed_list:
        print('|   {}   |      {}      |    {:.4f}    |'.format(i, result_dict[i][0], result_dict[i][1]))
        print('+--------+-------------+--------------+')
        
        