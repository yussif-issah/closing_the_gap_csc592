import torch


class PsjaModelInterface:

    def __init__(self, model_interface):

        self.model_interface = model_interface

        self.models = model_interface.models
        self.bounds = (0, 1)
        self.model_calls = 0

        self.new_adversarial_def = False

    def __call__(self, images, vanilla=False):
        return self.get_probs(images, vanilla)

    def get_probs(self, images, vanilla=False):
        # transform images for PSJA format - in InfoMax for t_map == 1: we have to unsqueeze, to make sure that the permutaiton still works
        if images.ndim == 3:
            images = images.unsqueeze(dim=0)

        images = images.permute(0, 3, 1, 2)
        return self.model_interface.get_probs(images, vanilla)

    def sample_bernoulli(self, probs):
        self.model_calls += probs.numel()
        return torch.bernoulli(probs)

    def decision(self, batch, label, num_queries=1, targeted=False):
        N = batch.shape[0] * num_queries
        self.model_calls += batch.shape[0] * num_queries
        # if N <= 100*1000:
        if batch.ndim == 3:
            new_batch = batch.repeat(num_queries, 1, 1)
        else:
            new_batch = batch.repeat(num_queries, 1, 1, 1)
        decisions = self._decision(new_batch, label, targeted)
        decisions = decisions.view(-1, len(batch)).transpose(0, 1)
        # elif num_queries <= 100*1000:
        #     decisions = torch.zeros(len(batch), num_queries, device=batch.device)
        #     for b in range(len(batch)):
        #         if batch.ndim == 3:
        #             new_batch = batch[b].view(-1, 1, 1).repeat(num_queries, 1, 1)
        #         else:
        #             new_batch = batch[b].view(-1, 1, 1, 1).repeat(num_queries, 1, 1, 1)
        #         decisions[b] = self._decision(new_batch, label, targeted)
        # else:
        #     decisions = torch.zeros(len(batch), num_queries, device=batch.device)
        #     for q in range(num_queries):
        #         decisions[:, q] = self._decision(batch, label, targeted)
        return decisions

    def _decision(self, batch, label, targeted=False):
        """
        :param label: True/Targeted labels of the original image being attacked
        :param num_queries: Number of times to query each image
        :param batch: A batch of images
        :param targeted: if targeted is true, label=targeted_label else label=true_label
        :return: decisions of shape = (len(batch), num_queries)
        """
        probs = self.get_probs(images=batch)
        prediction = probs.argmax(dim=1)
        if targeted:
            return (prediction == label) * 1.0
        else:
            return (prediction != label) * 1.0
