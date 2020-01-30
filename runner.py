import argparse
import os
import subprocess
import json

# generate benchmarking cases
def gen_cases(params):
    global cases
    if len(params) == 0:
        return cases
    prev_lenght = len(cases)
    param_name = list(params.keys())[0]
    n_param_values = len(params[param_name])
    cases = cases * n_param_values
    for i in range(n_param_values):
        for j in range(prev_lenght):
            cases[prev_lenght * i + j] += ' {}{} {}'.format(
                '-' if len(param_name) == 1 else '--',
                param_name, params[param_name][i])
    del params[param_name]
    gen_cases(params)


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='ConfigPath', type=str,
                    default='config.json')
parser.add_argument('--output', metavar='OutputPath', type=str,
                    default='stdout')
parser.add_argument('--dummy-run', default=False, action='store_true')
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)

os.system("mkdir -p data")

result = {}
# get CPU information
lscpu_info = subprocess.run(
    ['lscpu'], stdout=subprocess.PIPE, encoding='utf-8').stdout
while '  ' in lscpu_info:
    lscpu_info = lscpu_info.replace('  ',' ') # remove additional spaces
lscpu_info = lscpu_info.split('\n')
if '' in lscpu_info:
    lscpu_info.remove('')
for i in range(len(lscpu_info)):
    lscpu_info[i] = lscpu_info[i].split(': ')
result['HW'] = {'CPU': {line[0]: line[1] for line in lscpu_info}}

log = ''
stderr_file = open('_stderr.log', 'w')

common_params = config['common']
for params_set in config['cases']:
    cases = ['']
    params = common_params.copy()
    params.update(params_set.copy())
    algorithm = params['algorithm']
    libs = params['lib']
    del params['dataset'], params['algorithm'], params['lib']
    gen_cases(params)
    print('\n{} algorithm: {} case(s)\n'.format(algorithm, len(cases)))
    for dataset in params_set['dataset']:
        if dataset['training'].startswith('synth'):
            if dataset['training'].startswith('synth_reg'):
                _, _, train_samples, features = dataset['training'].split('_')
                x_train_file = 'data/synth-reg-X-train-{}x{}.npy'.format(
                    train_samples, features)
                y_train_file = 'data/synth-reg-y-train-{}x{}.npy'.format(
                    train_samples, features)
                paths = '--file-X-train {} --file-y-train {}'.format(
                    x_train_file, y_train_file)
                if 'testing' in dataset.keys():
                    _, _, test_samples, _ = dataset['testing'].split('_')
                    x_test_file = 'data/synth-reg-X-test-{}x{}.npy'.format(
                        test_samples, features)
                    y_test_file = 'data/synth-reg-y-test-{}x{}.npy'.format(
                        test_samples, features)
                    paths += ' --file-X-test {} --file-y-test {}'.format(
                        x_test_file, y_test_file)
                    command = 'python make_datasets.py -s {} -ts {} -f {} regression -x {} -y {} -xt {} -yt {}'.format(
                        int(train_samples)+int(test_samples),
                        test_samples, features, x_train_file, y_train_file,
                        x_test_file, y_test_file)
                else:
                    command = 'python make_datasets.py -s {} -f {} regression -x {} -y {}'.format(
                        train_samples, features, x_train_file, y_train_file)
                if not os.path.isfile(x_train_file) and not args.dummy_run:
                    print(command)
                    os.system(command)
            elif dataset['training'].startswith('synth_cls'):
                _, _, classes, train_samples, features = dataset['training'].split('_')
                x_train_file = 'data/synth-cls-{}-X-train-{}x{}.npy'.format(
                    classes, train_samples, features)
                y_train_file = 'data/synth-cls-{}-y-train-{}x{}.npy'.format(
                    classes, train_samples, features)
                paths = '--file-X-train {} --file-y-train {}'.format(
                    x_train_file, y_train_file)
                if 'testing' in dataset.keys():
                    _, _, classes, test_samples, _ = dataset['testing'].split('_')
                    x_test_file = 'data/synth-cls-{}-X-test-{}x{}.npy'.format(
                        classes, test_samples, features)
                    y_test_file = 'data/synth-cls-{}-y-test-{}x{}.npy'.format(
                        classes, test_samples, features)
                    paths += ' --file-X-test {} --file-y-test {}'.format(
                        x_test_file, y_test_file)
                    command = 'python make_datasets.py -s {} -ts {} -f {} classification -c {} -x {} -y {} -xt {} -yt {}'.format(
                        int(train_samples)+int(test_samples),
                        test_samples, features, classes,
                        x_train_file, y_train_file,
                        x_test_file, y_test_file)
                else:
                    command = 'python make_datasets.py -s {} -f {} classification -c {} -x {} -y {}'.format(
                        train_samples, features, classes,
                        x_train_file, y_train_file)
                if not os.path.isfile(x_train_file) and not args.dummy_run:
                    print(command)
                    os.system(command)
            else:
                raise ValueError('Unknown dataset type')
        else:
            raise ValueError(
                'Unknown dataset. Only synthetics are supported now')
        for lib in libs:
            for i, case in enumerate(cases):
                command = 'python {}/{}.py --output-format json{} {}'.format(
                    lib, algorithm, case, paths)
                print(command)
                if not args.dummy_run:
                    r = subprocess.run(
                        command.split(' '), stdout=subprocess.PIPE,
                        stderr=stderr_file, encoding='utf-8')
                    log += r.stdout

while '}\n{' in log:
    log = log.replace('}\n{', '},\n{')

log = '{"results":[\n' + log + '\n]}'

result.update(json.loads(log))
result = json.dumps(result, indent=4)

if args.output == 'stdout':
    print(result, end='\n')
else:
    with open(args.output, 'w') as output:
        output.write(result)
stderr_file.close()
