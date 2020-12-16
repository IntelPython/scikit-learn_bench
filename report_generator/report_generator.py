import openpyxl
import argparse
import json
import hashlib
from string import ascii_uppercase
import datetime


def get_property(entry, prop):
    keys = prop.split(':')
    value = entry
    for key in keys:
        value = value[key]
    return value


def result_entries_have_same_values(first_entry, second_entry, props,
                                    error_on_missing=True):
    res = True
    for prop in props:
        try:
            res = res and \
                (get_property(first_entry, prop) == get_property(second_entry, prop))
        except KeyError:
            if error_on_missing:
                raise KeyError()
    return res


def result_entries_are_equal(first_entry, second_entry, config):
    props = config['align'] + config['diff']
    return result_entries_have_same_values(first_entry, second_entry, props, True)


def result_entries_are_comparable(first_entry, second_entry, config):
    props = config['align']
    return result_entries_have_same_values(first_entry, second_entry, props, False)


def result_entries_have_same_diff(first_entry, second_entry, config):
    props = config['diff']
    return result_entries_have_same_values(first_entry, second_entry, props, False)


def results_are_mergeable(first_res, second_res, merging):
    hw_hash_equality = first_res['hardware_hash'] == second_res['hardware_hash']
    sw_hash_equality = first_res['software_hash'] == second_res['software_hash']
    if merging == 'hw_only':
        return hw_hash_equality
    elif merging == 'sw_only':
        return sw_hash_equality
    else:
        return sw_hash_equality and hw_hash_equality


excel_header_columns = list(ascii_uppercase)
for sym1 in ascii_uppercase:
    for sym2 in ascii_uppercase:
        excel_header_columns.append(sym1 + sym2)


def xy_to_excel_cell(x, y):
    return '{}{}'.format(excel_header_columns[x], y + 1)


def write_cell(work_sheet, x, y, value):
    work_sheet[xy_to_excel_cell(x, y)] = value


def create_list(res_entry, props_list):
    line = []
    for prop in props_list:
        try:
            val = get_property(res_entry, prop)
        except:
            val = ''
        line.append(val)
    return line


parser = argparse.ArgumentParser()
parser.add_argument('--result-files', type=str, required=True,
                    help='Benchmark result file names separated by commas')
parser.add_argument('--report-file', type=str,
                    default=f'report_{str(datetime.date.today())}.xlsx')
parser.add_argument('--generation-config', type=str,
                    default='default_report_gen_config.json')
parser.add_argument('--merging', type=str, default='none',
                    choices=('full', 'none', 'sw_only', 'hw_only'))
args = parser.parse_args()

json_results = []
for file_name in args.result_files.split(','):
    with open(file_name, 'r') as file:
        json_results.append(json.load(file))

with open(args.generation_config, 'r') as file:
    gen_config = json.load(file)

wb = openpyxl.Workbook()

# compute hash for software and hardware configurations
HASH_LIMIT = 8
for i, json_res in enumerate(json_results):
    for ware in ['software', 'hardware']:
        h = hashlib.sha256()
        h.update(bytes(str(json_res[ware]), encoding='utf-8'))
        json_res[f'{ware}_hash'] = h.hexdigest()[:HASH_LIMIT]

# create list of all result entry from all json logs
all_res_entries = []
for i, json_res in enumerate(json_results):
    extra_entry_info = json_res.copy()
    del extra_entry_info['results']
    for res_entry in json_res['results']:
        new_res_entry = res_entry.copy()
        new_res_entry.update(extra_entry_info)
        all_res_entries.append(new_res_entry)

if args.merging != 'none':
    for i, resi_entry in enumerate(all_res_entries):
        already_exist = False
        for j, resj_entry in enumerate(all_res_entries):
            if i == j or resi_entry == {} or resj_entry == {}:
                continue
            if result_entries_are_equal(resi_entry, resj_entry, gen_config):
                if resi_entry['measurement_time'] < resj_entry['measurement_time']:
                    resi_entry = resj_entry
                    resj_entry = {}

while {} in all_res_entries:
    all_res_entries.remove({})

diff_combinations = []
for i, res_entry in enumerate(all_res_entries):
    already_exist = False
    for diff_comb in diff_combinations:
        if result_entries_have_same_diff(res_entry, diff_comb, gen_config):
            already_exist = True
            break
    if not already_exist:
        diff_comb = res_entry.copy()
        diff_combinations.append(diff_comb)

align_combinations = []
for i, res_entry in enumerate(all_res_entries):
    already_exist = False
    for align_comb in align_combinations:
        if result_entries_are_comparable(res_entry, align_comb, gen_config):
            already_exist = True
            break
    if not already_exist:
        align_comb = res_entry.copy()
        align_combinations.append(align_comb)

HEAD_OFFSET = len(gen_config['diff'])
LEFT_OFFSET = len(gen_config['align'])

stages_splitter = {
    'training': ['training', 'computation'],
    'inference': ['prediction', 'transformation', 'search']
}

for stage_key in stages_splitter.keys():
    ws = wb.create_sheet(title=f'Results ({stage_key})')

    for i, col in enumerate(gen_config['align'] + ['time, s']):
        write_cell(ws, i, HEAD_OFFSET, col)

    for i, row in enumerate(gen_config['diff']):
        write_cell(ws, LEFT_OFFSET - 1, i, row)

    stage_align_combinations = align_combinations.copy()

    for align_comb in stage_align_combinations:
        if align_comb['stage'] not in stages_splitter[stage_key]:
            stage_align_combinations.remove(align_comb)

    for i, align_comb in enumerate(stage_align_combinations):
        arr = create_list(align_comb, gen_config['align'])
        for j, el in enumerate(arr):
            write_cell(ws, j, HEAD_OFFSET + 1 + i, el)

    for i, diff_comb in enumerate(diff_combinations):
        arr = create_list(diff_comb, gen_config['diff'])
        for j, el in enumerate(arr):
            write_cell(ws, LEFT_OFFSET + i, j, el)

    for i, res_entry in enumerate(all_res_entries):
        if res_entry['stage'] not in stages_splitter[stage_key]:
            continue
        x, y = None, None
        for j, align_comb in enumerate(stage_align_combinations):
            if result_entries_are_comparable(res_entry, align_comb, gen_config):
                y = j
                break
        for j, diff_comb in enumerate(diff_combinations):
            if result_entries_have_same_diff(res_entry, diff_comb, gen_config):
                x = j
                break
        write_cell(ws, LEFT_OFFSET + x, HEAD_OFFSET + 1 + y, res_entry['time[s]'])

# write configs
for i, json_res in enumerate(json_results):
    ws = wb.create_sheet(title=f"SW config n{i}_{json_res['software_hash']}")
    ws[xy_to_excel_cell(0, 0)] = \
        f"Software configuration {i} (hash: {json_res['software_hash']})"
    sw_conf = json.dumps(json_res['software'], indent=4).split('\n')
    for j in range(len(sw_conf)):
        ws[xy_to_excel_cell(0, 1 + j)] = sw_conf[j]

    ws = wb.create_sheet(title=f"HW config n{i}_{json_res['hardware_hash']}")
    ws[xy_to_excel_cell(0, 0)] = \
        f"Hardware configuration {i} (hash: {json_res['hardware_hash']})"
    hw_conf = json.dumps(json_res['hardware'], indent=4).split('\n')
    for j in range(len(hw_conf)):
        ws[xy_to_excel_cell(0, 1 + j)] = hw_conf[j]

wb.save(args.report_file)
