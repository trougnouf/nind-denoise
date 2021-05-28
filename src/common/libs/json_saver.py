# -*- coding: utf-8 -*-

from typing import Optional, Set
import sys
sys.path.append('..')
from common.libs import utilities

class JSONSaver:
    def __init__(self, jsonfpath, step_type: str = ['step', 'epoch'][0],
                 default={'best_val': dict()}):
        self.best_key_str = 'best_{}'.format(step_type)  # best step/epoch #
        self.jsonfpath = jsonfpath
        self.results = utilities.jsonfpath_load(jsonfpath, default=default)
        if self.best_key_str not in self.results:
            self.results[self.best_key_str] = dict()

    def add_res(self, step: int, res: dict, minimize=True, write=True,
                val_type=float, epoch=None, rm_none=False, key_prefix=''):
        '''epoch is an alias for step
        Set rm_none to True to ignore zero values'''
        if epoch is not None and step is None:
            step = epoch
        elif (epoch is None and step is None) or step is None or epoch is not None:
            raise ValueError('JSONSaver.add_res: Must specify either step or epoch')
        if step not in self.results:
            self.results[step] = dict()
        if key_prefix != '':
            res_ = dict()
            for akey, aval in res.values():
                res_[key_prefix+akey] = aval
            res = res_
        for akey, aval in res.items():
            if val_type is not None:
                aval = val_type(aval)
            self.results[step][akey] = aval
            if isinstance(aval, list):
                continue
            if rm_none and aval == 0:
                continue
            if akey not in self.results['best_val'] and akey in self.results[self.best_key_str]:  # works when best_val has been removed but best_step exists
                self.results['best_val'][akey] = self.results[self.results[self.best_key_str][akey]][akey]
            if (akey not in self.results[self.best_key_str]
                or akey not in self.results['best_val']
                or (self.results['best_val'][akey] > aval and minimize)
                or (self.results['best_val'][akey] < aval and not minimize)):
                self.results[self.best_key_str][akey] = step
                self.results['best_val'][akey] = aval
        if write:
            utilities.dict_to_json(self.results, self.jsonfpath)

    def write(self):
        utilities.dict_to_json(self.results, self.jsonfpath)
        
    def get_best_steps(self) -> Set[int]:
        return set(self.results[self.best_key_str].values())