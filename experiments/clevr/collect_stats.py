import os

def collect_report(report_path):
    losses = []
    for line in open(report_path, 'r'):
        if "Test" in line:
            info = line.split()
            epoch_id = int(info[2][:-1])
            avg_loss = float(info[5][:-1])
            avg_accu = float(info[8][1:-3]) / 100
            losses.append(str(avg_accu))
    print('\n'.join(losses))


if __name__ == "__main__":
    report_name = "all_rela_model_0.0001_topbotk_5_dim_12544_bw_False_obj_5_train_10000_val_100_model_2_latent_1024"
    report_path = os.path.join("/home/jianih/research/scallop-v2/experiments/data/CLEVR/models/", report_name)
    collect_report(report_path)
