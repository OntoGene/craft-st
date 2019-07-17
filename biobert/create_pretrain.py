import time 
import glob 
import os

def get_label_set_pretrain(NUM_LABELS_JO=None,LABEL_SET_PRETRAIN=None):
        
        #ontology = FLAGS.onto
        # ontology = 'PR'
        # path_to_data = '/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll/' + ontology + '.1000.conll'

        # tag_list = []
        # with open(path_to_data, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         try:
        #             tag_list.append(line.split('\t')[1].rstrip())
        #         except:
        #             pass
        #     tag_set = set(tag_list)
        
        # print(tag_list)

        # print(tag_set, '\ntag set-', len(tag_set), '\ntag list-', len(tag_list))
        


                
        f_path = '/mnt/storage/scratch1/jocorn/craft/biobert/data/pretrain/conll/'
        all_files = glob.glob(os.path.join(f_path, "*.conll"))

        for conll_file in all_files:
            onto = conll_file.split('/')[-1].split('.')[0]
            offi_set_size = conll_file.split('/')[-1].split('.')[1]
            out_dir_path = '/'.join(conll_file.split('/')[:-1])
            print (onto, ' : ',offi_set_size)

            ontology = onto.rstrip()
            tag_list = []
            with open(conll_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        tag_list.append(line.split('\t')[1].rstrip())
                    except:
                        pass
                tag_set = set(tag_list)
            
            #print(tag_list)

            real_set_size = len(tag_set)
            print('real tag set-', real_set_size, '\ntag list-', len(tag_list))
            

            out_file_path = out_dir_path + '/tag_set' + '_' + ontology + '.'+ offi_set_size + '-' + str(real_set_size) + '.txt'
            print(out_file_path)
            with open(out_file_path, 'w', encoding='utf-8') as out_f:
                for item in tag_set:
                    out_f.write(str(item) + "\n")
                pass
            print('\n\n')


        #tag_set.append("X")
        #tag_set.append("[CLS]")
        #tag_set.append("[SEP]")
        
        #print('after', len(tag_set), tag_set,'\n\n\n\n\n')

        # print('\n{:{align}{width}}'.format('*' * 60, align='^', width='80'))
        # print('{:{align}{width}}'.format( ontology , align='^', width='80'))
        # print('{:{width}}{}: {}'.format(' ', 'NUM_LABELS_JO', NUM_LABELS_JO, width='20'))
        # print('{:{width}}{}: {}'.format(' ', 'PATH TO DATA', path_to_data, width='20'))
        # print('{:{align}{width}}'.format('*' * 60 +'\n', align='^', width='80'))
        time.sleep(5.5)    

get_label_set_pretrain()