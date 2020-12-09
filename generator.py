from BnsLib.network.generators import FileHandeler, MultiFileHandeler, FileGenerator
from BnsLib.types import DataSize, LimitedSizeDict
import h5py
import numpy as np

FileGenerator = FileGenerator

def xyz_of_sky(sky_coords):
    dec = sky_coords[1]
    ra = sky_coords[0]
    return [np.cos(ra) * np.cos(dec),
            np.sin(ra) * np.cos(dec),
            np.sin(dec)]

std_bounds = [(35, 80), (35, 80), (1000, 3000), (0.15, 0.35), None,
              (0, np.pi), None, None, None]

class SignalHandler(FileHandeler):
    def __init__(self, file_path, sky_coords=[7, 8], ignore=[4, 6],
                 ref_index=0, cache=True, size_limit=None, mass=[0, 1],
                 bounds=std_bounds):
        """
        Arguments
        ---------
        file_path : str
            The path to the file that contains the data.
        sky_coords : {list of int, [7, 8]}
            A list of the indices within `x_data` that contains the
            sky-coordinates of the system (in radians). [ra, dec]
        ignore : {list of int, [4, 6]}
            A list of indices of `x_data`. The values corresponding to
            these indices will not be passed to the network.
        ref_index : {int, 0}
            This reference index is subtracted from the input index to
            give the index of the file.
        cache : {bool, True}
            Whether or not to cache the signals that were fetched
            before. Increases memory usage but decreases loading times
            significantly.
        size_limit : {DataSize or None, None}
            A size-limit (in binary units, e.g. MB) for the cache. If
            set to None, no size-limit will be enforced and the system
            may crash when it runs out of memory. To set a size-limit
            use BnsLib.types.DataSize.
        mass : {list of int, [0, 1]}
            The indices of the masses. The generator will assure that
            the bigger mass will get the lower index. Leave this list
            empty if you don't want masses to be sorted.
        """
        super().__init__(file_path)
        self.file = None
        self.sky_coords = sky_coords
        self.mass = mass
        self.ignore = ignore
        self.ref_index = ref_index
        self.bounds = bounds
        if cache:
            if size_limit is None:
                self.cache = {}
            else:
                self.cache = LimitedSizeDict(size_limit=size_limit,
                                             error_on_overflow=False)
        else:
            self.cache = None
    
    def __len__(self):
        try:
            return len(self.file['y_data_noisefree'])
        except:
            with h5py.File(self.file_path, 'r') as fp:
                ret = len(fp['y_data_noisefree'])
            return ret
    
    def file_index(self, index):
        return index - self.ref_index
    
    def __contains__(self, index):
        index = self.file_index(index)
        if index < 0:
            return False
        def contain_helper(ds, idx):
            return idx < len(ds)
        
        #Look in cache if it exists
        if self.cache is not None:
            if index in self.cache:
                return True
        
        #If sample was not loaded from cache, load from file
        try:
            return contain_helper(self.file['y_data_noisefree'], index)
        except:
            with h5py.File(self.file_path, 'r') as fp:
                ret = contain_helper(fp['y_data_noisefree'], index)
            return ret
    
    def open(self, mode='r'):
        self.file = h5py.File(self.file_path, mode)
    
    def close(self):
        self.file.close()
    
    def __getitem__(self, index):
        if index not in self:
            raise IndexError
        index = self.file_index(index)
        
        #Load cache if exists
        ret = None
        fetched_cache = False
        if self.cache is not None:
            if index in self.cache:
                ret = self.cache[index]
                fetched_cache = True
        
        #If index was not available in the cache, load it from disk
        if ret is None:
            try:
                label_vals = self.file['x_data'][index]
                wav = self.file['y_data_noisefree'][index]
            except:
                with h5py.File(self.file_path, 'r') as fp:
                    label_vals = fp['x_data'][index]
                    wav = fp['y_data_noisefree'][index]
            
            labels = []
            #Rescale the parameters to range [0, 1]
            if self.bounds is not None:
                tmp = []
                for i in range(len(label_vals)):
                    if self.bounds[i] is None:
                        tmp.append(label_vals[i])
                    else:
                        tmp.append((label_vals[i] - min(self.bounds[i])) / (max(self.bounds[i]) - min(self.bounds[i])))
                label_vals = np.array(tmp)
            
            #Assert m1 > m2
            if len(self.mass) == 2:
                if label_vals[self.mass[0]] < label_vals[self.mass[1]]:
                    tmp = label_vals.copy()
                    tmp[self.mass[0]] = label_vals[self.mass[1]]
                    tmp[self.mass[1]] = label_vals[self.mass[0]]
                    label_vals = tmp
            
            #Transform (ra, dec) -> (x, y, z)
            sky_coords = [0, 0]
            for i, val in enumerate(label_vals):
                if i in self.ignore:
                    continue
                if i == self.sky_coords[0]:
                    sky_coords[0] = val
                elif i == self.sky_coords[1]:
                    sky_coords[1] = val
                else:
                    labels.append(val)
            xyz = np.array(xyz_of_sky(sky_coords))
            xyz = xyz / np.linalg.norm(xyz)
            labels.extend(list(xyz))
            ret = [np.expand_dims(np.array(labels), axis=0),
                   np.expand_dims(wav.T, axis=0)]
        
        #Cache samples that were loaded from disk, if requested
        if self.cache is not None and not fetched_cache:
            self.cache[index] = ret
        
        return ret

class NoiseHandler(FileHandeler):
    def __init__(self, sample_rate=256., num_channels=3):
        super().__init__('')
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def __contains__(self, index):
        return True
    
    def open(self, mode='r'):
        return
    
    def close(self):
        return
    
    def __getitem__(self, index):
        shape = (1, int(self.sample_rate), int(self.num_channels))
        return np.random.normal(loc=0, scale=np.sqrt(self.sample_rate / 2),
                                size=shape)

class MultiHandler(MultiFileHandeler):
    def __init__(self, input_shape=[(8,), (256, 3)], output_shape=(8,)):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.add_file_handeler(NoiseHandler(), group='noise')
    
    def __len__(self):
        ret = 0
        for fh in self.file_handeler_groups['signal']:
            ret += len(fh)
        return ret
    
    def split_index_to_groups(self, index):
        return {'signal': index,
                'noise': True}
    
    def format_return(self, inp):
        labels, signal = inp['signal']
        noise = inp['noise']
        total = signal + noise
        mean = np.expand_dims(np.mean(total, axis=-1), axis=-1)
        std = np.expand_dims(np.std(total, axis=-1), axis=-1)
        total = (total - mean) / std
        return [labels, total], labels
    
    def open(self, mode='r'):
        for fh in self.file_handelers:
            fh.open(mode=mode)
    
    def close(self):
        for fh in self.file_handelers:
            fh.close()
