import argparse
import os
from train import train
from test import test
from eval import eval


if __name__== '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'mode')

    # yolov5
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--img_dir", help="Set in-image_path")
    train_parser.add_argument("--label_dir", help="Set in-image path")
    train_parser.add_argument('--img-size', nargs='+', type=int, default = [512, 512, 3], help='model input size for training')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    train_parser.add_argument('--class_num', type=int, help='total class number including background')
    train_parser.add_argument("--class_weights", action='store_true', help="compute class_weights while training", dest="class_weights")
    train_parser.add_argument("--model_dir", help="Set out model path", default="./weights/")
    train_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--img_dir", help="Set in-image path")
    test_parser.add_argument("--model_path", help="Set trained model path")
    test_parser.add_argument('--class_num', type=int, help='total class number including background')
    test_parser.add_argument('--img-size', nargs='+', type=int, default = [512, 512, 3], help='model input size for training')
    test_parser.add_argument("--save_dir", help="Set out image path", default="dataset/result")
    test_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    val_parser = subparsers.add_parser("eval")
    val_parser.add_argument("--img_dir", help="Set in-image path")
    val_parser.add_argument("--label_dir", help="Set in-image path")
    val_parser.add_argument("--model_path", help="Set trained model path")
    val_parser.add_argument('--class_num', type=int, help='total class number including background')
    val_parser.add_argument('--img-size', nargs='+', type=int, default = [512, 512, 3], help='model input size for training')
    val_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    parser_args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu

    if parser_args.mode == 'train':
        train(parser_args)
    elif parser_args.mode == 'test':
        test(parser_args)
    elif parser_args.mode == 'eval':
        eval(parser_args)
        pass
