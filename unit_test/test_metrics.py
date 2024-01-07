import pytest
import torch

from ..src.metrics import Metrics

class TestMetrics:
    
   def setup_method(self):
      self.utterance_values = torch.FloatTensor(1000, 1).uniform_(1, 5)
      self.system_ids = torch.randint(22, 58, (1000, 1))
      self.mos_scores = torch.randint(1, 5, (1000, 1))

      self.metrics = Metrics('metrics_test')

   def teardown_method(self):
      self.metrics.clear()

   def _add_data(self, batch_size):
      for ind, _  in enumerate(self.utterance_values[::batch_size]):
         start = ind * batch_size
         end = (ind+1) * batch_size
         self.metrics.update(
            self.utterance_values[start:end], 
            self.system_ids[start:end], self.mos_scores[start:end]
         )
      
   def test_can_update_values(self):
      self._add_data(6)
      assert True

   def test_can_print_values(self):
      self._add_data(6)
      self.metrics.print()
      self.metrics.print(1)
      assert True
      
   def test_can_save_values(self):
      pass

   def test_can_clear_values(self):
      pass