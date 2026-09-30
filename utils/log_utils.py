import csv


class CsvLogger:
    def __init__(self, path):
        self.path = path
        self.header = []

    def log(self, row, step=None):
        current_row = dict(row)
        if step is not None:
            current_row['step'] = step

        if len(self.header) == 0:
            self.header = list(current_row.keys())

        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header, extrasaction='ignore')
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(current_row)

    def close(self):
        return None
