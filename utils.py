from continuum.tasks.task_set import TaskSet
import torch

def get_samples_based_on_importance(taskset:TaskSet, correct_ex_indices, predictions_all, device):
    with torch.no_grad():
        y = torch.tensor(taskset._y).to(device)
        preds = predictions_all[correct_ex_indices]
        v = torch.topk(preds, 2, dim=2)[0]
        abs = torch.abs(torch.diff(v))
        flattened = torch.flatten(abs)
        indices_sorted = torch.sort(flattened)[1]
        indices_sorted = indices_sorted.cpu().numpy()
    
        # indices_sorted = indices_sorted * (task_id +1)
        items = taskset.get_raw_samples(indices_sorted)

        return items

def count_parameters(model, verbose=True):
        '''
        Count number of parameters, print to screen.
        '''
        total_params = learnable_params = fixed_params = 0
        for param in model.parameters():
            n_params = index_dims = 0
            for dim in param.size():
                n_params = dim if index_dims==0 else n_params*dim
                index_dims += 1
            total_params += n_params
            if param.requires_grad:
                learnable_params += n_params
            else:
                fixed_params += n_params
        if verbose:
            print("--> this network has {} parameters (~{} million)"
                .format(total_params, round(total_params / 1000000, 1)))
            print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                        round(learnable_params / 1000000, 1)))
            print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
        return total_params, learnable_params, fixed_params
  