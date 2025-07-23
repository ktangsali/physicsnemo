import re
import matplotlib.pyplot as plt
import argparse
import os

# Regular expressions to match the relevant lines in the log file
epoch_pattern = re.compile(r'epoch (\d+):')
iteration_pattern = re.compile(r'batch processed: (\d+)')
loss_header_pattern = re.compile(r'^([a-zA-Z_]+(\s+[a-zA-Z_]+)+)\s*$')
loss_value_pattern = re.compile(r'(\d\.\d+e[+-]\d+)')

def parse_log_file(filepath):
    total_iterations = []
    losses = {}
    loss_keys = []
    with open(filepath, 'r') as file:
        current_epoch = None
        for line in file:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            iteration_match = iteration_pattern.search(line)
            if iteration_match and current_epoch is not None:
                batch_number = int(iteration_match.group(1))
                total_iteration = current_epoch * 55 + batch_number  # Adjust as needed
                total_iterations.append(total_iteration)
                continue

            if not loss_keys and "batch processed" not in line and "Device" not in line:
                # Only consider lines with multiple words and no numbers as header
                if re.match(r'^(\s*[a-zA-Z_]+\s+)+[a-zA-Z_]+\s*$', line):
                    potential_keys = line.strip().split()
                    loss_keys = potential_keys
                    for key in loss_keys:
                        losses[key] = []
                continue

            loss_values = loss_value_pattern.findall(line)
            if loss_values:
                for key, value in zip(loss_keys, loss_values):
                    losses[key].append(float(value))
    return total_iterations, losses

def get_colors(n, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N) for i in range(n)]

def main():
    parser = argparse.ArgumentParser(description='Plot losses from multiple log files.')
    parser.add_argument('logfiles', nargs='+', help='Paths to log files to parse and plot')
    parser.add_argument('--output', default='multi_log_plot.png', help='Output plot filename')
    args = parser.parse_args()

    all_keys = set()
    log_data = {}
    for logfile in args.logfiles:
        print(logfile)
        iterations, losses = parse_log_file(logfile)
        log_data[logfile] = {'iterations': iterations, 'losses': losses}
        print(log_data[logfile]["losses"].keys())
        all_keys.update(losses.keys())

    all_keys = sorted(all_keys)
    num_losses = len(all_keys)
    rows = (num_losses + 2) // 3
    # print(all_keys)
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle('Losses per Mini Batch (Multiple Logs)')

    colors = get_colors(len(log_data.keys()))
    colors_dict = {}
    for i, logfile in enumerate(log_data.keys()):
        colors_dict[logfile] = colors[i]

    for idx, key in enumerate(all_keys):
        row, col = divmod(idx, 3)
        ax = axs[row, col] if rows > 1 else axs[col]
        for logfile, data in log_data.items():
            if key in data['losses']:
                label = os.path.basename(logfile)
                # Ensure matching lengths for iterations and losses
                y = data['losses'][key]
                x = data['iterations'][:len(y)]
                y = y[:len(x)]
                # print(len(y), len(x))
                ax.plot(x, y, color=colors_dict[logfile], label=label)
        ax.set_title(key)
        ax.set_xlabel('Total Iteration')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()

    # Hide any unused subplots
    for idx in range(len(all_keys), rows * 3):
        row, col = divmod(idx, 3)
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    main()
