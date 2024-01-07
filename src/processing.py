import os
import pandas as pd
from datetime import datetime
from scipy.stats import wasserstein_distance


def prepare_somos_data(
    listeners: int,
    systems: int,
    texts: int,
    num_iterations: int,
    somos_dir: str = 'data/SOMOS',
    output_dir: str = 'data/somos_prepared',
    random_seed: int = 42
):
    '''
        prepares somos data for training and validation, saving the results to a seperate directory
        the files are labelled <<output_dir>>/validation_set.csv and <<output_dir>>/test_set.csv
    '''
    full_data_path = os.path.join(somos_dir, 'raw_scores_with_metadata/raw_scores.tsv')
    full_data = pd.read_csv(full_data_path, delimiter='\t')
    full_data['utteranceId'] = full_data.utteranceId + '.wav'
    full_data = full_data.drop_duplicates(subset=['listenerId', 'utteranceId', 'choice', 'systemId'])

    # remove all listeners who evaluated more than 3000 samples 
    listener_val_counts = full_data.listenerId.value_counts()
    valid_listener_ids = pd.Series(listener_val_counts[listener_val_counts < 3000].index)

    full_data = full_data[full_data.listenerId.isin(valid_listener_ids)]


    def prepare_scores(filename: str):
        '''
            function to prepare either the validation or test sets, minimizing the wasserstein distance between the sampled set and the full data of the split
        '''

        scores_path = os.path.join(somos_dir, f'training_files/split1/full/{filename}')
        scores = pd.read_csv(scores_path)

        scores = scores[scores.listenerId.isin(valid_listener_ids)]

        full_scores = scores.merge(full_data, how='left', on=['listenerId', 'utteranceId', 'choice', 'systemId'])

        iterations = 0
        repetitions = 0
        min_wasser_distance = float('inf')
        result_df = None

        while iterations < num_iterations:
            try:
                repetitions += 1

                state = random_seed + repetitions
                # take x listeners at random
                listener_sample = valid_listener_ids.sample(listeners, random_state=state*10)
                iteration_df = full_scores[full_scores.listenerId.isin(listener_sample)]
                
                # take y systems that the listeners evaluated
                system_sample = iteration_df.systemId.value_counts()\
                    .sample(systems, random_state=(state+1)*10).index
                iteration_df = iteration_df[iteration_df.systemId.isin(system_sample)]

                # take z texts that the listeners evaluated and the systems predicted
                text_sample = iteration_df.sentenceId.value_counts()\
                    .sample(texts, random_state=(state+2)*10).index
                iteration_df = iteration_df[iteration_df.sentenceId.isin(text_sample)]

                w_distance = wasserstein_distance(full_scores.choice, iteration_df.choice)
                if w_distance < min_wasser_distance:
                    min_wasser_distance = w_distance
                    result_df = iteration_df

                iterations += 1
            except ValueError:
                continue

        return result_df, min_wasser_distance
        

    # prepare validation and test sets
    validation_df, w_distance = prepare_scores('VALIDSET')
    print(f'{datetime.now()} :: VALIDATION SET PREPARED :: {len(validation_df)} SAMPLES')
    print(f'VAL min wasserstein distance: {w_distance}')
    print('validation locales')
    for locale_name, num_locale in validation_df.locale.value_counts().items():
        print(f'{locale_name}: {num_locale}')

    test_df, w_distance = prepare_scores('TESTSET')
    print(f'{datetime.now()} :: TEST SET PREPARED :: {len(test_df)} SAMPLES')
    print(f'TEST min wasserstein distance: {w_distance}')
    print('test locales')
    for locale_name, num_locale in test_df.locale.value_counts().items():
        print(f'{locale_name}: {num_locale}')

    # save validation and test sets in <<output_dir>>
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    validation_df.to_csv(os.path.join(output_dir, 'validation_set.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    print(f'{datetime.now()} :: SAVED TO {output_dir}')