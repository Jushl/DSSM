class TwoStageDecoupledStreamingSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_info = []
        global_offset = 0
        for data in dataset.data_structure:
            self.data_info.append({
                'start_idx': global_offset,
                'end_idx': global_offset + data['num'],
                'num': data['num']
            })
            global_offset += data['num']

        self.num_data = len(self.data_info)
        self.total_frames = global_offset

    def __iter__(self):
        current_positions = [0] * self.num_data
        current_batch_indices = list(range(min(self.batch_size, self.num_data)))
        while len(current_batch_indices) >= self.batch_size:
            batch_indices = []
            new_batch_indices = []
            for i in current_batch_indices:
                if current_positions[i] < self.data_info[i]['num']:
                    absolute_idx = self.data_info[i]['start_idx'] + current_positions[i]
                    batch_indices.append(absolute_idx)
                    current_positions[i] += 1
                    new_batch_indices.append(i)
                else:
                    next_video = self._find_next_available_video(current_positions, max(i,
                                                                                        max(current_batch_indices)) if current_batch_indices else 0)
                    if next_video is not None and next_video not in new_batch_indices:
                        absolute_idx = self.data_info[next_video]['start_idx'] + current_positions[next_video]
                        batch_indices.append(absolute_idx)
                        current_positions[next_video] += 1
                        new_batch_indices.append(next_video)

            while len(batch_indices) < self.batch_size:
                next_video = self._find_next_available_video(current_positions,
                                                             max(new_batch_indices) if new_batch_indices else 0)
                if next_video is None or next_video in new_batch_indices:
                    break
                absolute_idx = self.data_info[next_video]['start_idx'] + current_positions[next_video]
                batch_indices.append(absolute_idx)
                current_positions[next_video] += 1
                new_batch_indices.append(next_video)

            if len(batch_indices) == self.batch_size:
                yield batch_indices

            current_batch_indices = new_batch_indices
            available_videos = [i for i in range(self.num_data) if current_positions[i] < self.data_info[i]['num']]
            if len(available_videos) < self.batch_size:
                break

    def _find_next_available_video(self, current_positions, start_index):
        for i in range(start_index + 1, self.num_data):
            if current_positions[i] < self.data_info[i]['num']:
                return i
        for i in range(0, start_index):
            if current_positions[i] < self.data_info[i]['num']:
                return i
        return None

    def __len__(self):
        total_batches = 0
        current_positions = [0] * self.num_data
        current_batch_indices = list(range(min(self.batch_size, self.num_data)))
        while len(current_batch_indices) >= self.batch_size:
            batch_size_count = 0
            new_batch_indices = []

            for i in current_batch_indices:
                if current_positions[i] < self.data_info[i]['num']:
                    batch_size_count += 1
                    current_positions[i] += 1
                    new_batch_indices.append(i)
                else:
                    next_video = self._find_next_available_video(current_positions, max(i,
                                                                                        max(current_batch_indices)) if current_batch_indices else 0)
                    if next_video is not None and next_video not in new_batch_indices:
                        batch_size_count += 1
                        current_positions[next_video] += 1
                        new_batch_indices.append(next_video)

            while batch_size_count < self.batch_size:
                next_video = self._find_next_available_video(current_positions,
                                                             max(new_batch_indices) if new_batch_indices else 0)
                if next_video is None or next_video in new_batch_indices:
                    break
                batch_size_count += 1
                current_positions[next_video] += 1
                new_batch_indices.append(next_video)
            if batch_size_count == self.batch_size:
                total_batches += 1
            current_batch_indices = new_batch_indices
            available_videos = [i for i in range(self.num_data) if current_positions[i] < self.data_info[i]['num']]
            if len(available_videos) < self.batch_size:
                break
        return total_batches
