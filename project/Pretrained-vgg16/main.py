from run_cross_validation import run_cross_validation, run_single
from nets.net import baseline, vgg_std16_model
from utils.preprocessing import data_generator

HEIGHT, WIDTH = 48, 64
NUM_FEATURES = HEIGHT * WIDTH
BATCH_SIZE = 32
EPOCHS = 1
K_FOLDS = 2

if __name__ == '__main__':
    # Baseline line model
    # run_cross_validation(img_rows=HEIGHT, img_cols=WIDTH, batch_size=BATCH_SIZE,
    #                      create_model=baseline,
    #                      nfolds=K_FOLDS, to_submit=True,
    #                      nb_epoch=EPOCHS, read_from_cache=True, save=True, model_name='baseline_48x64_color')

    ## Baseline with data rotation, and shift (augmentation)
    # run_cross_validation(img_rows=HEIGHT, img_cols=WIDTH, batch_size=BATCH_SIZE,
    #                      create_model=baseline,
    #                      nfolds=K_FOLDS, to_submit=False,
    #                      nb_epoch=EPOCHS, read_from_cache=True, save=True,
    #                      model_name='baseline_48x64_rotation_shift_color', augmentation=data_generator)


    ### Run this only for testing purposes!
    # run_single(img_rows=HEIGHT, img_cols=WIDTH, batch_size=BATCH_SIZE,
    #                      create_model=baseline,
    #                      nfolds=K_FOLDS, to_submit=False,
    #                      nb_epoch=EPOCHS, read_from_cache=True, save=False, model_name='baseline_48x64_color_test')

    # VGG-16 pretrained run
    # run_single(img_rows=224, img_cols=224, batch_size=BATCH_SIZE, color=True,
    #                      create_model=vgg_std16_model,
    #                      nfolds=K_FOLDS, to_submit=False,
    #                      nb_epoch=EPOCHS, read_from_cache=False, save=False, model_name='vgg-16-pretrained')

    # Run vgg-16 with 5 fold CV.
    run_cross_validation(nfolds=5, img_rows=224, img_cols=224, batch_size=BATCH_SIZE, color=True,
           create_model=vgg_std16_model, to_submit=False,
           nb_epoch=EPOCHS, read_from_cache=False, save=False, model_name='vgg-16-pretrained')