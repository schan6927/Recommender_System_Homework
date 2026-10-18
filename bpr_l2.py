from recbole.model.general_recommender.bpr import BPR

class BPRL2(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.reg_lambda = float(config['reg_lambda'])

    def calculate_loss(self, interaction):
        ranking_loss = super().calculate_loss(interaction)
        user = self.user_embedding(interaction[self.USER_ID])
        positive = self.item_embedding(interaction[self.ITEM_ID])
        negative = self.item_embedding(interaction[self.NEG_ITEM_ID])
        squared_norm = (user.square().sum(dim=1)
                        + positive.square().sum(dim=1)
                        + negative.square().sum(dim=1))
        regularization = squared_norm.mean() / 2
        return ranking_loss + self.reg_lambda * regularization
