import argparse


valid_models = ["hybrid", "att", "tf_encoder", "penta"]
valid_embeddings = ["none", "glove", "bert"]



def init_argparse():
    """
    CLI Arguments for training phase 
    """
    parser = argparse.ArgumentParser(
    prog="Training Model",
    usage="Arguments: --model, --embedding, --batch_size(optional).",
    description="""Example: python train.py --model penta --embedding None --dataset_size 1 --batch_size 32 -- epochs 3
                 --model: Type of model for training. Expect one of ['hybrid', 'att', 'tf_encoder', 'penta'].
                 --embedding: Type of word-level embedding. Expect one of [None, 'glove', 'bert'].
                 --dataset_size: Dataset size for training. Default: 1 (All dataset).
                 --batch_size: Batch size. Default: 32.
                 --epochs: Epochs. Default: 3."""
)
    parser.add_argument("--model", required=True, help='Type of model: hybrid, att, tf_encoder, penta')
    parser.add_argument(
        "--embedding", required=True,
        help='Word embedding: None, Glove or BERT'
    )
    parser.add_argument(
        "--dataset_size", required=False, default= 1, help= "Dataset size"
    )
    parser.add_argument(
        "--batch_size",required=False ,default = 32,
        help='Batch size'
    )
    parser.add_argument(
        "--epochs",required=False ,default = 3,
        help='Epochs'
    )
    return parser



def init_infer_argparse():
    """
    CLI Arguments for infer phase
    """
    parser = argparse.ArgumentParser(
    prog="Training Model",
    usage="Arguments: --model, --embedding",
    description="""Example: python infer.py --model penta --embedding None 
                 --model: Type of model for training. Expect one of ['hybrid', 'att', 'tf_encoder', 'penta'].
                 --embedding: Type of word-level embedding. Expect one of [None, 'glove', 'bert'].
                 """
)
    parser.add_argument("--model", required=True, help='Type of model: hybrid, att, tf_encoder, penta')
    parser.add_argument(
        "--embedding", required=True,
        help='Word embedding: None, Glove or BERT'
    )
    return parser


def check_valid_args(args):
    # Check valid model type
    if str(args.model).lower() not in valid_models:
        raise TypeError("No model named: {}, expected valid model belongs to {}".format(args.model, valid_models))
    elif str(args.embedding).lower() not in valid_embeddings:
        raise TypeError("No embedding type named: {}, expeted valid embedding belongs to {}".format(args.embedding, valid_embeddings))
    
    return True


if __name__ == "__main__":
    parser = init_argparse()
    args   = parser.parse_args()