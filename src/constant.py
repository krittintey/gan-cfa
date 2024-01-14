import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 5e-5
epochs = 3
train_batch_size = 32
valid_batch_size = 32
test_batch_size = 32
unlabeled_batch_size = 32
class_num = 4
bidirectional = True

# Define hyperparameters
train_params = {'batch_size': train_batch_size,
                #'shuffle':True,
                'num_workers':0
               }
valid_params = {'batch_size':valid_batch_size,
                'shuffle':False,
                'num_workers':0
               }
test_params = {'batch_size':test_batch_size,
                'shuffle':False,
                'num_workers':0
               }
unlabeled_params = {'batch_size':unlabeled_batch_size,
                    'shuffle':True,
                    'num_workers':0
                   }
