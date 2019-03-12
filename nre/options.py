"""
    Implementation of all availabel options.
    model_opts: used for construction of the model
    preprocess_opts: used for data preprocess
"""


def model_opts(parser):
    """
    used for constructing the model
    """

    # Embedding options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-word_vec_size', type=int, default=50, help='Word embedding size of words')
    group.add_argument('-vocabulary_size', type=int, default=114042, help='Vocabulary size of corpus')
    group.add_argument('-position_size', type=int, default=5, help='Position embedding size')
    group.add_argument('-type_range', type=int, default=4, help='The number of all types')
    group.add_argument('-type_size', type=int, default=5, help='Type embedding size')

    # Layers options
    group = parser.add_argument_group('Model-Layers')
    group.add_argument('-cnn_window_size', type=int, default=3, help='The filter width of CNN')
    group.add_argument('-hidden_size', type=int, default=230, help='The hidden embedding size of sentences')
    group.add_argument('-num_classes', type=int, default=53, help='The number of all relation classes')
    group.add_argument('-model_type', type=str, default='PDCNN+TATT', help='Name of used model, e.g. cnn+att, dcnn+att, pdcnn+att')
    group.add_argument('-alpha', type=float, default=0.8, help='Weight of type logits trained by model')
    group.add_argument('-beta', type=float, default=0.2, help='Weight of the loss from type classification')

    # Entity type options
    group = parser.add_argument_group('Model-Entity')
    group.add_argument('-use_type_embedding', type=bool, default=False, help='Whether use entity type or not')

    # Train options
    group = parser.add_argument_group('Model-Train')
    group.add_argument('-dropout_keep', type=float, default=0.5, help='Dropout rate')
    group.add_argument('-learning_rate', type=float, default=0.25, help='Learning rate')
    group.add_argument('-learning_rate_decay', type=float, default=0.5, help='for Adagrad')
    group.add_argument('-weight_decay', type=float, default=1e-5, help='weight decay')
    group.add_argument('-optimizer', type=str, default='sgd', help='Optimizer for training')
    group.add_argument('-batch_size', type=int, default=16, help='Batch size')
    group.add_argument('-train_steps', type=int, default=50000, help='Total number of training steps')
    group.add_argument('-check_steps', type=int, default=2000, help='Steps for evaluation')
    group.add_argument('-print_steps', type=int, default=100, help='print log info every print_steps')
    group.add_argument('-pretrain_model', type=str, default='model_step_best.pt', help='Load existed model')
    group.add_argument('-stop_after_n_eval', type=int, default=10, help='Stop training when the performance has not improved during n evaluation times')
    group.add_argument('-debug_mode', type=bool, default=False, help='If debug, only evaluate one batch')

    # GPU options
    group = parser.add_argument_group('Model-GPU')
    group.add_argument('-gpu_ranks', default=[0,1,2,4], nargs='+', type=int, help='list of ranks of each process')
    group.add_argument('-gpu_num', default=0, type=int, help='total number of distributed processed')
    group.add_argument('-gpu_master', default=None, type=int, help='Device id when call cuda()')

    # Misc options
    group = parser.add_argument_group('Model-Misc') 
    group.add_argument('-model_prefix', type=str, default='ckpt/model_step_', help='Model prefix for saving models')
    group.add_argument('-model_max_num', type=int, default=5, help='Max number of checkpoints')
    group.add_argument('-checkpoints_history_file', type=str, default='ckpt/history.pt', help='A list containing file names of checkpoints')
    group.add_argument('-log_file', type=str, default='', help='Log file for logging')
    group.add_argument('-summary_dir', type=str, default='', help='Directory for storing summaries')
    group.add_argument('-res_dir', type=str, default='ckpt', help='Directory for storing results')


def preprocess_opts(parser):
    """
    Used for preprocessing the original data
    """

    # File options
    group = parser.add_argument_group('Preprocess-File')
    group.add_argument('-word2vec_file', type=str, default='raw/vec.txt', help='Pretrained word2vec file')
    group.add_argument('-relation2id_file', type=str, default='raw/relation2id.txt', help='Relation2id map file')
    group.add_argument('-type2id_file', type=str, default='raw/type2id.txt', help='Type2id map file')
    group.add_argument('-train_file', type=str, default='raw/train.txt', help='Training set')
    group.add_argument('-test_file', type=str, default='raw/test.txt', help='Testing set')
    group.add_argument('-tmp_train_bag', type=str, default='raw/tmp_train.txt', help='temp train file for gathering bag')
    group.add_argument('-tmp_test_bag', type=str, default='raw/tmp_test.txt', help='temp test file for gathering bag')


def shared_opts(parser):
    """
    Used for preprocessing and training
    """

    group = parser.add_argument_group('Shared-Options')
    group.add_argument('-position_num', type=int, default=201, help='The number of position embeddings')
    group.add_argument('-num_steps', type=int, default=120, help='The fixed sentence length')
    group.add_argument('-type_num', type=int, default=3, help='The number of entity types except NA')
    # There are too many processed files, their names all constrainted to the format of '[train|test]_variable.pt' except the vec.pt
    group.add_argument('-processed_dir', type=str, default='data', help='Directory for storing processed data')

def plot_opts(parser):
    """
    Used for ploting pr curves
    """
    group = parser.add_argument_group('Plot-Options')
    group.add_argument('-pr_data_dir', type=str, default='pr_data', help='Directory for storing precision and recall data')
    group.add_argument('-xlim', type=float, default=0.4, help='The recall ranges from 0 to xlim')
    group.add_argument('-ylim', type=float, default=0.3, help='The precision ranges from ylim to 1.0')
