from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss, accuracy_score
from keras.callbacks import EarlyStopping

from utils.load import read_and_normalize_train_data, read_and_normalize_test_data,\
    copy_selected_drivers, dict_to_list, merge_several_folds_mean, create_submission, save_model, read_model

import numpy as np
'''
Run nfold cross validation.
NOTE: make sure to pass model creation function
'''
def run_cross_validation(nfolds=10, color=True, img_rows=24, img_cols=32, batch_size=32, create_model=None, nb_epoch=1,
                         read_from_cache=False, to_submit=False, save=True, model_name='baseline', augmentation=None):
    if create_model is None:
        raise Exception('create_model function is not passed!')

    random_state = 51
    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_train_data(img_rows, img_cols, color, cache=read_from_cache)

    if to_submit:
        test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color, cache=read_from_cache)

    if augmentation is not None:
        augmentation.fit(train_data)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]

    for train_drivers, test_drivers in kf:
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        model = create_model(img_rows, img_cols, color)

        if augmentation is not None:
            history = model.fit_generator(augmentation.flow(X_train, Y_train, batch_size=batch_size),
                                          samples_per_epoch=X_train.shape[0], nb_epoch=nb_epoch,
                                          validation_data=(X_valid, Y_valid),
                                          callbacks=callbacks
                                          )
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                                verbose=1, validation_data=(X_valid, Y_valid),
                                callbacks=callbacks)
        predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        if to_submit:
            # Store test predictions
            test_prediction = model.predict(test_data, batch_size=128, verbose=1)
            yfull_test.append(test_prediction)

    score = log_loss(train_target, dict_to_list(yfull_train))
    _, accuracy = model.evaluate(train_target, dict_to_list(yfull_train), verbose=1)

    print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    info_string = 'loss_' + str(score) \
                  + '_accuracy_' + str(accuracy) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_folds_' + str(nfolds) \
                    + '_ep_' + str(nb_epoch)

    if to_submit:
        test_res = merge_several_folds_mean(yfull_test, nfolds)
        create_submission(test_res, test_id, info_string)
    if save:
        save_model(model, model_name)


def run_single(nfolds=10, color=True, img_rows=24, img_cols=32, batch_size=32, create_model=None, nb_epoch=1,
                         read_from_cache=False, to_submit=False, save=True, model_name='baseline', augmentation=None):
    random_state = 51
    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_train_data(img_rows, img_cols, color=color, cache=read_from_cache)
    #test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color, cache=read_from_cache)

    yfull_test = []
    unique_list_train = ['p002', 'p012']#, 'p014', 'p015', 'p016', 'p021', 'p022', 'p035', 'p041', 'p042', 'p045', 'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p081']
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p024']#, 'p026', 'p039', 'p072']
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print('Start Single Run')
    print('Split train: ', len(X_train))
    print('Split valid: ', len(X_valid))
    print('Train drivers: ', unique_list_train)
    print('Valid drivers: ', unique_list_valid)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]
    model = create_model(img_rows, img_cols, color=color)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    score = log_loss(Y_valid, predictions_valid)

    print('Score log_loss:', score)

    # Store test predictions
    # test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
    # yfull_test.append(test_prediction)

    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))
    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_ep_' + str(nb_epoch)

    full_pred = model.predict(train_data, batch_size=batch_size, verbose=1)
    score = log_loss(train_target, full_pred)
    _, accuracy = model.evaluate(train_target, full_pred, verbose=1)
    print('Full score log_loss: {}, accuracy: {} '.format(score, accuracy))

    # test_res = merge_several_folds_mean(yfull_test, 1)
    # create_submission(model_name, test_res, test_id, info_string)

    #save_useful_data(full_pred, train_id, model, info_string)
    # https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_cv_drivers_v2.py

if __name__ == '__main__':
    pass
