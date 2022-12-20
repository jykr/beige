from dataclasses import dataclass
from typing import Iterable, Literal, Dict, Tuple
from xmlrpc.client import Boolean
import torch
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import get_alpha0
import get_pi_alpha0

@dataclass
class NData:
    def __init__(self, 
        adata: ad.AnnData, 
        n_sgRNAs_per_target: int,
        sorting:Literal["topbot", "bins"], 
        actualize = True, 
        repguide_mask: str = None, 
        dispersion_path:str = None, 
        fit_a0 = False, 
        device:str = None, 
        sample_mask_column:str = None, 
        tiling: bool=False, 
        pi_popt: Tuple[float] = None,
        fit_a0_lower_quantile: float = None
        ):
        """Initialize the NData for normal model from ReporterScreen object.

        Args:
            X_mask: Binary mask for invalid data points. 1 for valid and 0 for invalid data points. Should have the same shape as adata.X
            sample_mask_column: Column indicating the mask for low-quality samples. 1 for valid and 0 for invalid samples.
            fit_a0: If True, estimate alpha0 parameter for DirichletMultinomial overdispersion model.
            tiling: if True, variables only there for variant screens are not initialized.
            pi_popt: popt fitted from external dataset can be provided to estimate pi_a0 instead of fitting log_a0 ~ log_n relationship.
            fit_a0_lower_quantile: if not None, only the data points with lowest `fit_a0_lower_quantile` residuals will be used to fit the log_a0 ~ log_n relationship.
            repguide_mask: key of `adata.uns` that stores pd.DataFrame of replicate x guide mask
        """
        if adata.shape[1] == 0: return None
        self.device=device
        if actualize:
            self.sorting_scheme = sorting
            adata.condit["size_factor_bcmatch"] = self.get_size_factor(adata.layers["X_bcmatch"])
            adata.condit["size_factor"] = self.get_size_factor(adata.X)
            if not ("rep" in adata.condit.columns and "sort" in adata.condit.columns):
                adata.condit["rep"], adata.condit["sort"] = zip(*adata.condit.index.map(lambda s: s.rsplit("_", 1)))
            adata.condit["rep_id"] = -1
            for i, rep in enumerate(sorted(adata.condit.rep.unique())):
                adata.var.loc[adata.condit.rep == rep, "rep_id"] = i
            adata = adata[:, adata.condit.sort_values("rep_id").index]
            adata.var = adata.var.reset_index(drop=True)
            if not 'sort' in adata.condit.columns:
                adata.condit['sort'] = adata.condit['index'].map(lambda s:s.split("_")[-1])
            adata.var = adata.condit
            self.adata_sorted = adata[:,adata.condit.sort != "bulk"]
            self.adata_bulk = adata[:, adata.condit.sort == "bulk"]

            self.n_samples = len(adata.condit)    #8
            self.n_guides = len(adata.guides)
            self.n_reps = len(adata.condit.rep.unique())
            self.n_bins = len(self.adata_sorted.var.sort.unique()) # excluding bulk
            if not tiling:
                self.n_targets = len(adata.guides.target.unique())
                self.n_sgRNAs_per_target = n_sgRNAs_per_target  #5

            if sorting == "topbot":
                self.adata_sorted.condit["sort_id"] = 0
                self.adata_sorted.condit.loc[self.adata_sorted.condit.sort == "bot", "sort_id"] = 1
                self.upper_bounds = torch.tensor([1.0, 0.3])
                self.lower_bounds = torch.tensor([0.7, 0.0])
                
            elif sorting == "bins":
                self.adata_sorted.condit["sort_id"] = 0
                self.adata_sorted.condit.loc[self.adata_sorted.condit.sort == "high", "sort_id"] = 1
                self.adata_sorted.condit.loc[self.adata_sorted.condit.sort == "low", "sort_id"] = 2
                self.adata_sorted.condit.loc[self.adata_sorted.condit.sort == "bot", "sort_id"] = 3
                self.upper_bounds = torch.tensor([1.0, 0.8, 0.4, 0.2])
                self.lower_bounds = torch.tensor([0.8, 0.6, 0.2, 0.0])

            if dispersion_path is None:
                self.dispersion = None
            else: 
                dispersion= pd.read_csv(dispersion_path, header=None).values
                assert len(dispersion) == self.n_guides
                self.dispersion = dispersion

            self.adata_sorted = self.adata_sorted[:, self.adata_sorted.condit.sort_values(["rep_id", "sort_id"]).index]
            self.adata_sorted.var = self.adata_sorted.var.reset_index(drop=True)

            # Make sure the adata are sorted
            if not sample_mask_column is None:
                self.sample_mask = torch.as_tensor(self.adata_sorted.condit[sample_mask_column].to_numpy()).reshape(self.n_reps, self.n_bins)
                self.bulk_sample_mask = torch.as_tensor(self.adata_bulk.condit[sample_mask_column].to_numpy())
            else:
                self.sample_mask = torch.ones((self.n_reps, self.n_bins), dtype=Boolean)
                self.bulk_sample_mask = torch.ones(self.n_reps, dtype=Boolean)
            if not self.device is None:
                self.sample_mask = self.sample_mask.cuda()
                self.bulk_sample_mask = self.bulk_sample_mask.cuda()

            self.X = self.transform_data(self.adata_sorted.X)   # (n_reps, n_bins, n_guides)
            self.X_masked = self.X * self.sample_mask[:,:,None]
            self.X_bcmatch = self.transform_data(self.adata_sorted.layers["X_bcmatch"]) 
            self.X_bcmatch_masked = self.X_bcmatch * self.sample_mask[:,:,None]
            self.X_bulk = self.transform_data(self.adata_bulk.X, 1)
            self.X_bcmatch_bulk = self.transform_data(self.adata_bulk.layers["X_bcmatch"], 1)
            self.size_factor_bcmatch = torch.as_tensor(self.adata_sorted.condit["size_factor_bcmatch"].to_numpy()).reshape(
                self.n_reps, self.n_bins
            )
            self.size_factor = torch.as_tensor(self.adata_sorted.condit["size_factor"].to_numpy()).reshape(
                self.n_reps, self.n_bins
            )
            self.size_factor_bcmatch_bulk = torch.as_tensor(self.adata_bulk.condit["size_factor_bcmatch"].to_numpy()).reshape(
                self.n_reps, 1
            )
            self.size_factor_bulk = torch.as_tensor(self.adata_bulk.condit["size_factor"].to_numpy()).reshape(
                self.n_reps, 1
            )

            if not self.device is None:
                self.size_factor = self.size_factor.cuda()
                self.size_factor_bcmatch = self.size_factor_bcmatch.cuda()

            if repguide_mask is None:
                #self.repguide_mask = torch.ones((self.n_reps, self.n_guides)) > 0
                self.repguide_mask = ~(self.X == 0).any(axis=1)
            else:
                print(f"Using repguide mask {repguide_mask}")
                assert repguide_mask in adata.uns.keys(), f"{repguide_mask} not in adata.uns"
                assert self.adata_sorted.uns[repguide_mask].shape == (self.n_guides, self.n_reps)
                assert (self.adata_sorted.uns[repguide_mask].index == self.adata_sorted.guides.index).all()
                assert (self.adata_sorted.uns[repguide_mask].columns == self.adata_sorted.condit.rep.unique()).all()
                self.repguide_mask = torch.as_tensor(self.adata_sorted.uns[repguide_mask].values.T) > 0
                self.repguide_mask = torch.logical_and(self.repguide_mask, ~(self.X == 0).any(axis=1))

            if not tiling: 
                edited_bulk = self.transform_data(self.adata_bulk.layers['edits'], n_bins = 1)
                nonedited_bulk = self.X_bcmatch_bulk - edited_bulk
                assert (nonedited_bulk >= 0).all()
                self.allele_counts_bulk = torch.stack([edited_bulk, nonedited_bulk], axis=-1) # (n_reps, n_bins, n_guides, n_alleles)
                assert (self.allele_counts_bulk.sum(axis=-1) == self.X_bcmatch_bulk).all()
                self.target_lengths = self.get_target_lengths(adata)
                edited = self.transform_data(self.adata_sorted.layers['edits'])
                nonedited = self.X_bcmatch - edited
                self.allele_counts = torch.stack([edited, nonedited], axis=-1) # (n_reps, n_bins, n_guides, n_alleles)
            if not pi_popt is None:
                pi_a0 = get_pi_alpha0.get_pred_alpha0(
                    self.allele_counts_bulk.clone().cpu(), 
                    self.size_factor_bulk.clone().cpu(), 
                    pi_popt
                )
            else:
                pi_a0, _pi_popt = get_pi_alpha0.get_fitted_alpha0(
                    self.allele_counts_bulk.clone().cpu(), 
                    self.size_factor_bulk.clone().cpu(), 
                    fit_quantile=fit_a0_lower_quantile
                )
            self.pi_a0 = torch.as_tensor(pi_a0)
            a0, popt = get_alpha0.get_fitted_alpha0(
                self.X.clone().cpu(), 
                self.size_factor.clone().cpu(), 
                fit=fit_a0, 
                fit_quantile=fit_a0_lower_quantile
            )
            self.a0 = torch.as_tensor(a0)
            a0_bcmatch = get_alpha0.get_pred_alpha0(
                self.X_bcmatch.clone().cpu(), 
                self.size_factor_bcmatch.clone().cpu(), 
                popt
            )
            self.a0_bcmatch = torch.as_tensor(a0_bcmatch)
            edited_a0 = torch.as_tensor(
                get_alpha0.get_pred_alpha0(
                    self.allele_counts[:,:,:,0].clone().cpu(), 
                    self.size_factor.clone().cpu(), 
                    popt
                )
            )
            unedited_a0 = torch.as_tensor(
                get_alpha0.get_pred_alpha0(
                    self.allele_counts[:,:,:,1].clone().cpu(), 
                    self.size_factor.clone().cpu(), 
                    popt
                )
            )
            self.a0_allele = torch.stack([edited_a0, unedited_a0], dim=-1).clamp(1e-5)
                #self.a0_bcmatch = torch.tensor(get_fitted_alpha0(self.X_bcmatch, self.size_factor_bcmatch))
            if not fit_a0: 
                a0, popt = get_alpha0.get_fitted_alpha0(
                    self.X.clone().cpu(), 
                    self.size_factor.clone().cpu(), 
                    fit=False, 
                    fit_quantile=fit_a0_lower_quantile
                )
                a0 = torch.as_tensor(a0)
                a0[torch.isnan(a0)] = self.a0[torch.isnan(a0)]
                self.a0 = a0
                try:
                    edited_a0 = torch.as_tensor(
                        get_alpha0.get_fitted_alpha0(
                            self.allele_counts[:,:,:,0].clone().cpu(), 
                            self.size_factor.clone().cpu(), 
                            fit = False, 
                            fit_quantile=fit_a0_lower_quantile
                            )[0]
                        )
                    unedited_a0 = torch.as_tensor(
                        get_alpha0.get_fitted_alpha0(
                            self.allele_counts[:,:,:,1].clone().cpu(), 
                            self.size_factor.clone().cpu(), 
                            fit = False, 
                            fit_quantile=fit_a0_lower_quantile
                            )[0]
                        )
                    pi_a0 = torch.as_tensor(
                        get_pi_alpha0.get_fitted_alpha0(
                            self.allele_counts_bulk.clone().cpu(), 
                            self.size_factor_bulk.clone().cpu(), 
                            fit=False,
                            fit_quantile=fit_a0_lower_quantile
                        )[0]
                    )
                    pi_a0[torch.isnan(pi_a0)] = self.pi_a0[torch.isnan(pi_a0)]
                    self.pi_a0 = pi_a0
                    
                except ValueError as e:
                    print(e)
                    edited_a0 = torch.as_tensor(
                        get_alpha0.get_pred_alpha0(
                            self.allele_counts[:,:,:,0].clone().cpu(), 
                            self.size_factor.clone().cpu(), 
                            fit = False, 
                            fit_quantile=fit_a0_lower_quantile
                            )[0]
                        )
                    unedited_a0 = torch.as_tensor(
                        get_alpha0.get_pred_alpha0(
                            self.allele_counts[:,:,:,1].clone().cpu(), 
                            self.size_factor.clone().cpu(), 
                            fit = False, 
                            fit_quantile=fit_a0_lower_quantile
                            )[0]
                        )
                a0_allele = torch.stack([edited_a0, unedited_a0], dim=-1)
                a0_allele[torch.isnan(a0_allele)] = self.a0_allele[torch.isnan(a0_allele)]
                self.a0_allele = a0_allele
                #self.a0_bcmatch = None

    def __getitem__(self, guide_idx):
        self.adata_sorted = self.adata_sorted[guide_idx,:]
        self.adata_bulk = self.adata_bulk[guide_idx,:]
        self.n_guides = len(guide_idx)
        if not self.n_targets is None:
            self.n_targets = len(self.adata_sorted.guides.target.unique())
        if not self.dispersion is None:
            self.dispersion = self.dispersion[guide_idx]
        self.X = self.X[:,:,guide_idx]   # (n_reps, n_bins, n_guides)
        self.X_masked = self.X_masked[:,:,guide_idx]
        self.X_bcmatch = self.X_bcmatch[:,:,guide_idx] 
        self.X_bcmatch_masked = self.X_bcmatch_masked[:,:,guide_idx]
        self.X_bulk = self.X_bulk[:,:,guide_idx]
        self.X_bcmatch_bulk = self.X_bcmatch_bulk[:,:,guide_idx]
        self.repguide_mask = self.repguide_mask[:,guide_idx]
        self.target_lengths = self.get_target_lengths(self.adata_sorted)
        self.allele_counts = self.allele_counts[:,:,guide_idx,:]
        self.a0 = self.a0[guide_idx]
        self.a0_allele = self.a0_allele[guide_idx, :]
        self.pi_a0 = self.pi_a0[guide_idx]
        
    @classmethod
    def makeempty(cls, adata):
        return(cls(adata, None, "topbot", None, actualize=False))

    def transform_data(self, X, n_bins = None):
        if n_bins is None: n_bins = self.n_bins
        x = torch.as_tensor(X).T.reshape(
            (self.n_reps, n_bins, self.n_guides)
        ).float()
        if not self.device is None:
            x = x.cuda()
        return x

    def get_size_factor(self, X:np.array):
        """
        Get size factor for samples.
        """
        n_guides, n_samples = X.shape
        geom_mean_x = np.exp((1/n_samples)*np.log(X+0.5).sum(axis=1))
        assert geom_mean_x.shape == (n_guides,)
        norm_count = X/geom_mean_x[:, None]
        size_factor = np.median(norm_count, axis = 0)
        assert size_factor.shape == (n_samples,)
        return(size_factor)

    def transform_allele(self, adata, collapsed_allele_df):
        # sort allele by (has_target, has_nontarget): (1,1)>(1,0)>(0,1)>(0,0)
        allele_tensor = torch.empty((self.n_reps, self.n_bins, self.n_guides, 4),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            for j in range(self.n_bins):
                condit_idx = np.where((adata.condit.rep_id == i) & (adata.condit.sort_id == j))[0]
                assert len(condit_idx) == 1, print(i, j, condit_idx)
                condit_idx = condit_idx.item()
                condit_allele_df = collapsed_allele_df.iloc[:,condit_idx].unstack(level=[1,2], fill_value = 0).astype(int)
                condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
                condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
                try:
                    assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, bin {}.".format(i, j)
                except:
                    return((condit_idx, condit_allele_df, condit_bcmatch_counts))
                condit_allele_df[(False, False)] = condit_bcmatch_counts - condit_allele_df.sum(axis=1)
                condit_allele_df = condit_allele_df.sort_values(["has_target", "has_nontarget"], axis=1, ascending=False)
                assert condit_allele_df.to_numpy().shape == (self.n_guides, 4)
                try:
                    allele_tensor[i, j, :, :] = torch.as_tensor(condit_allele_df.to_numpy())
                except:
                    return(allele_tensor, torch.as_tensor(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, collapsed_allele_df)
        return(allele_tensor)

    
    def transform_allele2(self, adata, collapsed_allele_df):
        # sort allele by (has_target, has_nontarget): (1,1)>(1,0)>(0,1)>(0,0)
        n_alleles = collapsed_allele_df.index.get_level_values('n_edits').max() + 1
        allele_tensor = torch.empty((self.n_reps, self.n_bins, self.n_guides, n_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            for j in range(self.n_bins):
                condit_idx = np.where((adata.condit.rep_id == i) & (adata.condit.sort_id == j))[0]
                assert len(condit_idx) == 1, print(i, j, condit_idx)
                condit_idx = condit_idx.item()
                condit_allele_df = collapsed_allele_df.iloc[:,condit_idx].unstack(level=1, fill_value = 0).astype(int)
                condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
                condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
                try:
                    assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, bin {}.".format(i, j)
                except:
                    return((condit_idx, condit_allele_df, condit_bcmatch_counts))
                condit_allele_df.loc[:,0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
                condit_allele_df = condit_allele_df.sort_values("n_edits", axis=1, ascending=True)
                assert condit_allele_df.to_numpy().shape == (self.n_guides, n_alleles)
                try:
                    allele_tensor[i, j, :, :] = torch.as_tensor(condit_allele_df.to_numpy())
                except:
                    return(allele_tensor, torch.as_tensor(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, collapsed_allele_df)
        return(allele_tensor)

    def transform_allele_bulk(self, adata, collapsed_allele_df):
        # sort allele by (has_target, has_nontarget): (1,1)>(1,0)>(0,1)>(0,0)
        allele_tensor = torch.empty((self.n_reps, 1, self.n_guides, 4),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            condit_idx = np.where(adata.condit.rep_id == i)[0]
            assert len(condit_idx) == 1, print(i, "bulk", condit_idx)
            condit_idx = condit_idx.item()
            condit_allele_df = collapsed_allele_df.iloc[:,condit_idx].unstack(level=[1,2], fill_value = 0).astype(int)
            condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
            condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
            try:
                assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, bulk.".format(i)
            except:
                return((condit_idx, condit_allele_df, condit_bcmatch_counts))
            condit_allele_df[(False, False)] = condit_bcmatch_counts - condit_allele_df.sum(axis=1)
            condit_allele_df = condit_allele_df.sort_values(["has_target", "has_nontarget"], axis=1, ascending=False)
            assert condit_allele_df.to_numpy().shape == (self.n_guides, 4)
            try:
                allele_tensor[i, 0, :, :] = torch.as_tensor(condit_allele_df.to_numpy())
            except:
                return(allele_tensor, torch.as_tensor(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, collapsed_allele_df)
        return(allele_tensor)

    def transform_allele_bulk2(self, adata, collapsed_allele_df):
        # sort allele by (has_target, has_nontarget): (1,1)>(1,0)>(0,1)>(0,0)
        n_alleles = collapsed_allele_df.index.get_level_values('n_edits').max() + 1
        allele_tensor = torch.empty((self.n_reps, 1, self.n_guides, n_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            condit_idx = np.where(adata.condit.rep_id == i)[0]
            assert len(condit_idx) == 1, print(i, "bulk", condit_idx)
            condit_idx = condit_idx.item()
            condit_allele_df = collapsed_allele_df.iloc[:,condit_idx].unstack(level=1, fill_value = 0).astype(int)
            condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
            condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
            try:
                assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {},.".format(i,)
            except:
                return((condit_idx, condit_allele_df, condit_bcmatch_counts))
            condit_allele_df.loc[:,0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
            condit_allele_df = condit_allele_df.sort_values("n_edits", axis=1, ascending=True)
            assert condit_allele_df.to_numpy().shape == (self.n_guides, n_alleles)
            try:
                allele_tensor[i, 0, :, :] = torch.as_tensor(condit_allele_df.to_numpy())
            except:
                return(allele_tensor, torch.as_tensor(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, collapsed_allele_df)
        return(allele_tensor)

    def get_target_lengths(self, adata):
        target_len_list = []
        adata.guides['target'] = adata.guides['target'].astype('category')
        cur_item= adata.guides['target'].cat.codes[0]
        cur_len = 0
        for i in adata.guides['target'].cat.codes:
            if cur_item == i:
                cur_len += 1
            else:
                target_len_list.append(cur_len)
                cur_len = 1
                cur_item = i
        target_len_list.append(cur_len)
        target_lengths = torch.tensor(target_len_list)
        return(target_lengths)

def _insert_row_(row_number: int, df: pd.DataFrame, row_value):
    # https://www.geeksforgeeks.org/insert-row-at-given-position-in-pandas-dataframe/
    # Function to insert row in the dataframe
    df1 = df[0:row_number]
    df2 = df[row_number:]
    df1.loc[row_number] = row_value
    df_result = pd.concat([df1, df2])
    df_result.index = [*range(df_result.shape[0])]
    return df_result


def _insert_row_to_obs(adata: ad.AnnData, row_number: int, row_value):
    obs = _insert_row_(row_number, adata.guides, row_value)
    X = np.insert(adata.X, row_number, 0, axis=0)
    layers = {}
    for layer in adata.layers.keys():
        layers[layer] = np.insert(adata.layers[layer], row_number, 0, axis=0)
    bdata = ad.AnnData(X, obs, adata.condit, adata.uns, adata.guidesm, adata.conditm, layers)
    return bdata


def check_consecutive_targets(target_list, guide_per_target_counts=5):
    item_count = 0
    prev_item = ""
    for item in target_list:
        if item_count == 0:
            item_count += 1
            prev_item = item
        else:
            if item == prev_item:
                item_count += 1
            else:
                assert (
                    item_count == guide_per_target_counts
                ), "not all targets are consecutive in len 5"
                item_count = 1
                prev_item = item


def standardize_guide_per_target_counts(adata: ad.AnnData, guide_per_target_counts=5):
    bdata = adata.copy()
    bdata.obs = bdata.obs.reset_index()
    bdata.obs.loc[:,"target"] = bdata.obs["name"].map(lambda s: s.rsplit("_", 1)[0])

    target_counts = (
        bdata.obs[["target", "Target gene/variant"]].groupby("target").count()
    )
    for target in target_counts.index[
        target_counts["Target gene/variant"] < guide_per_target_counts
    ]:
        for i in range(
            guide_per_target_counts - target_counts["Target gene/variant"][target]
        ):
            target_idx = np.where(bdata.obs.target == target)[0]
            target_column_idx = np.where("target" == bdata.obs.columns)[0].item()
            guide_column_idx = np.where("name" == bdata.obs.columns)[0].item()
            row_to_insert = [0.0] * len(bdata.obs.columns)
            row_to_insert[target_column_idx] = target
            row_to_insert[guide_column_idx] = "{}_gx{}".format(target, i)
            bdata = _insert_row_to_obs(bdata, max(target_idx), row_to_insert)
    check_consecutive_targets(bdata.obs.target.to_list())
    return bdata

def get_target_info(adata: ad.AnnData):
    adata.guides["target"] = adata.guides.index.map(lambda s: s.split("_g")[0])
    return adata.guides[["target", "Group"]].drop_duplicates()

def get_data(adata: ad.AnnData):
    adata = adata.copy()
    bulk_idx = np.where(adata.condit.index.map(lambda s: "bulk" in s))[0]
    adata.condit["depth"] = adata.X.sum(axis=0)
    adata.condit["depth_bcmatch"] = adata.layers["X_bcmatch"].sum(axis=0)
    adata.condit["condition"] = np.nan
    adata.condit.loc[adata.condit.sort == "top", "condition"] = +1
    adata.condit.loc[adata.condit.sort == "bulk", "condition"] = 0
    adata.condit.loc[adata.condit.sort == "bot", "condition"] = -1
    return(NData(adata, 5))

@dataclass
class MData(NData):
    '''
    Data structured for multiple alleles per guide.
    '''

    def __init__(self, adata, n_sgRNAs_per_target,
    sorting, plasmid_counts: np.ndarray, allele_df_key = 'aa_allele_counts'):
        super().__init__(adata, n_sgRNAs_per_target, sorting, set_variant=False)
        guide_to_allele, reindexed_df = self.reindex_allele_df(adata.uns[allele_df_key])
        self.n_max_alleles = reindexed_df.index.get_level_values('allele_id_for_guide').max() + 1 # include no edit allele
        
        self.edit_index = self.get_edit_to_index_dict(guide_to_allele)
        self.n_edits = len(self.edit_index.keys())
        
        self.allele_to_edit = self.get_allele_to_edit_tensor(adata, self.edit_index, guide_to_allele)
        assert self.allele_to_edit.shape == (self.n_guides, self.n_max_alleles-1, self.n_edits)

        self.allele_counts = self.transform_allele(self.adata_sorted, reindexed_df)
        self.allele_counts_bulk = self.transform_allele_bulk(self.adata_bulk, reindexed_df)
        self.allele_mask = self.get_allele_mask(adata, guide_to_allele)
        assert self.allele_counts.shape == (self.n_reps, self.n_bins, self.n_guides, self.n_max_alleles)
        assert self.allele_counts_bulk.shape == (self.n_reps, 1, self.n_guides, self.n_max_alleles)
        assert self.allele_mask.shape == (self.n_guides, self.n_max_alleles)



    
    def reindex_allele_df(self, alleles_df):
        '''
        Input: Dataframe of (guide, allele) -> (per sample count)
        Output: 
            - guide_allele_id_to_allele
                DataFrame of (guide, allele_id_for_guide) -> (global_allele_id, aa_allele(str))
            - reindexed_allele_df
                Dataframe of (guide, allele_id_for_guide) -> (per sample count)
            
            allele_id_for_guide: order of allele within a guide.
            global_allele_id: global unique id for each (guide, allele) pair.
        '''
        guide_to_allele = dict(list(alleles_df[['guide', 'aa_allele']].groupby('guide').aa_allele))
        dfs = []
        for k, s in guide_to_allele.items():
            df = s.reset_index()
            df = df.rename(columns={"index": "allele_id"})
            df.index= df.index+1
            df = df.reset_index()
            df = df.rename(columns={"index": "allele_id_for_guide"})
            df['guide'] = k
            dfs.append(df)
        guide_to_allele_tbl = pd.concat(dfs)
        alleles_df = pd.merge(alleles_df, guide_to_allele_tbl, on=["aa_allele", "guide"])
        reindexed_df = alleles_df.reset_index().set_index(['guide', 'allele_id_for_guide'])
        guide_allele_id_to_allele = reindexed_df[['index', 'aa_allele']]
        reindexed_allele_df = reindexed_df.drop(['aa_allele', 'index'], axis=1)
        return(guide_allele_id_to_allele, reindexed_allele_df)


    def transform_allele(self, adata, reindexed_df):
        '''
        Transform reindexed allele dataframe reindexed_df of (guide, allele_id_for_guide) -> (per sample count)
        to (n_reps, n_bins, n_guides, n_alleles) tensor.
        '''
        allele_tensor = torch.empty((self.n_reps, self.n_bins, self.n_guides, self.n_max_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            for j in range(self.n_bins):
                condit_idx = np.where((adata.condit.rep_id == i) & (adata.condit.sort_id == j))[0]
                assert len(condit_idx) == 1, print(i, j, condit_idx)
                condit_idx = condit_idx.item()
                condit_name = adata.condit.index[condit_idx]
                condit_allele_df = reindexed_df.loc[:,condit_name].unstack(level=1, fill_value = 0).astype(int)
                condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
                condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
                try:
                    assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, .".format(i, j)
                except Exception as e:
                    print(e)
                    return((condit_idx, condit_allele_df, condit_bcmatch_counts))
                condit_allele_df[0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
                condit_allele_df = condit_allele_df.sort_values("allele_id_for_guide", axis=1, ascending=True)
                assert all(condit_allele_df.columns == list(range(self.n_max_alleles)))
                assert condit_allele_df.to_numpy().shape == (self.n_guides, self.n_max_alleles)
                try:
                    allele_tensor[i, j, :, :] = torch.from_numpy(condit_allele_df.to_numpy())
                except:
                    return(allele_tensor, torch.from_numpy(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, reindexed_df)
        return(allele_tensor)


    def transform_allele_bulk(self, adata, reindexed_df):
        '''
        Transform reindexed allele dataframe reindexed_df of (guide, allele_id_for_guide) -> (per sample count)
        to (n_reps, n_bins, n_guides, n_alleles) tensor.
        '''
        
        allele_tensor = torch.empty((self.n_reps, 1, self.n_guides, self.n_max_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            condit_idx = np.where(adata.condit.rep_id == i)[0]
            assert len(condit_idx) == 1, print(i, "bulk", condit_idx)
            condit_idx = condit_idx.item()
            condit_name = adata.condit.index[condit_idx]
            condit_allele_df = reindexed_df.loc[:,condit_name].unstack(level=1, fill_value = 0).astype(int)
            condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
            condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
            try:
                assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, bin {}.".format(i,)
            except Exception as e:
                print(e)
                return((condit_idx, condit_allele_df, condit_bcmatch_counts))
            condit_allele_df[0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
            condit_allele_df = condit_allele_df.sort_values("allele_id_for_guide", axis=1, ascending=True)
            assert all(condit_allele_df.columns == list(range(self.n_max_alleles)))
            assert condit_allele_df.to_numpy().shape == (self.n_guides, self.n_max_alleles)
            try:
                allele_tensor[i, 0, :, :] = torch.from_numpy(condit_allele_df.to_numpy())
            except:
                return(allele_tensor, torch.from_numpy(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, reindexed_df)
        return(allele_tensor)


    def get_allele_mask(self, adata, guide_allele_id_to_allele_df):
        guide_to_allele_tbl = guide_allele_id_to_allele_df.reset_index()
        n_valid_allele = guide_to_allele_tbl.groupby('guide')['allele_id_for_guide'].max().reindex(adata.guides.index, fill_value=0)
        n_valid_allele[n_valid_allele == 0] = 1
        mask = torch.zeros((self.n_guides, self.n_max_alleles))
        for i in range(mask.shape[0]):
            mask[i, 0] = 1
            for j in range(n_valid_allele[i]):
                mask[i, j+1] = 1
        return mask.bool()

    def get_edit_to_index_dict(self, guide_allele_id_to_allele_df):
        '''
        Returns Dict[edit(str) -> index(int)] for all edits
        '''
        edit_lists = guide_allele_id_to_allele_df.aa_allele.map(lambda a: list(a.aa_allele.edits) + list(a.nt_allele.edits))
        edits = pd.Series(pd.Series([e.get_abs_edit() for l in edit_lists.tolist() for e in l]).unique())
        edits_to_index = pd.Series(edits)
        edits_to_index = edits_to_index.reset_index().rename(columns={0:'edit'}).set_index('edit')['index'].to_dict()
        return(edits_to_index)

    def get_allele_to_edit_tensor(self, adata, edits_to_index: Dict[str, int], guide_allele_id_to_allele_df: pd.DataFrame):
        guide_allele_id_to_allele_df['edits'] = guide_allele_id_to_allele_df.aa_allele.map(lambda a: list(a.aa_allele.edits) + list(a.nt_allele.edits))
        guide_allele_id_to_allele_df = guide_allele_id_to_allele_df.reset_index()
        guide_allele_id_to_allele_df['edit_idx'] = guide_allele_id_to_allele_df.edits.map(lambda es: [edits_to_index[e.get_abs_edit()] for e in es])
        guide_allele_id_to_edit_df = guide_allele_id_to_allele_df[['guide', 'allele_id_for_guide', 'edit_idx']].set_index(['guide', 'allele_id_for_guide'])
        guide_allele_id_to_edit_df = guide_allele_id_to_edit_df.unstack(level=1, fill_value=[]).reindex(
            adata.guides.index, fill_value=[])
        allele_edit_assignment = torch.zeros((len(adata.guides), self.n_max_alleles-1, len(edits_to_index.keys())))
        for i in range(len(guide_allele_id_to_edit_df)):
            for j in range(len(guide_allele_id_to_edit_df.columns)):
                allele_edit_assignment[i, j, guide_allele_id_to_edit_df.iloc[i,j]] = 1 
        return(allele_edit_assignment)



@dataclass
class VData(NData):
    '''
    Data structured for multiple alleles per guide- for variant tiling screen.
    '''

    def __init__(self, adata, n_sgRNAs_per_target, 
    sorting, fit_a0 = False, allele_df_key = 'allele_counts_filtered',
    remove_no_allele_guides = True):
        if adata.shape[1] == 0: return None
        if remove_no_allele_guides:
            n_alleles = adata.uns[allele_df_key][['guide', 'allele']].groupby('guide').count().reindex(adata.guides.index).allele
            adata = adata[n_alleles > 0,:]
        super().__init__(adata, n_sgRNAs_per_target, sorting, fit_a0=fit_a0)
        guide_to_allele, reindexed_df = self.reindex_allele_df(adata.uns[allele_df_key])
        self.guide_to_allele = guide_to_allele
        self.reindexed_df = reindexed_df
        self.n_max_alleles = reindexed_df.index.get_level_values('allele_id_for_guide').max() + 1 # include no edit allele

        self.edit_index = self.get_edit_to_index_dict(guide_to_allele)
        self.n_edits = len(self.edit_index.keys())
        self.allele_to_edit = self.get_allele_to_edit_tensor(adata, self.edit_index, guide_to_allele)
        assert self.allele_to_edit.shape == (self.n_guides, self.n_max_alleles-1, self.n_edits)

        self.allele_counts = self.transform_allele(self.adata_sorted, reindexed_df)
        self.allele_counts_bulk = self.transform_allele_bulk(self.adata_bulk, reindexed_df)
        self.allele_mask = self.get_allele_mask(adata, guide_to_allele)
        print("max # alleles: {}".format(self.n_max_alleles))
        assert self.allele_counts.shape == (self.n_reps, self.n_bins, self.n_guides, self.n_max_alleles)
        assert self.allele_counts_bulk.shape == (self.n_reps, 1, self.n_guides, self.n_max_alleles)
        assert self.allele_mask.shape == (self.n_guides, self.n_max_alleles)

    def reindex_allele_df(self, alleles_df):
        '''
        Input: Dataframe of (guide, allele) -> (per sample count)
        Output: 
            - guide_allele_id_to_allele
                DataFrame of (guide, allele_id_for_guide) -> (global_allele_id, aa_allele(str))
            - reindexed_allele_df
                Dataframe of (guide, allele_id_for_guide) -> (per sample count)
            
            allele_id_for_guide: order of allele within a guide.
            global_allele_id: global unique id for each (guide, allele) pair.
        '''
        guide_to_allele = dict(list(alleles_df[['guide', 'allele']].groupby('guide').allele))
        dfs = []
        for k, s in guide_to_allele.items():
            df = s.reset_index()
            df = df.rename(columns={"index": "allele_id"})
            df.index= df.index+1
            df = df.reset_index()
            df = df.rename(columns={"index": "allele_id_for_guide"})
            df['guide'] = k
            dfs.append(df)
        guide_to_allele_tbl = pd.concat(dfs)
        alleles_df = pd.merge(alleles_df, guide_to_allele_tbl, on=["allele", "guide"])
        reindexed_df = alleles_df.reset_index().set_index(['guide', 'allele_id_for_guide'])
        guide_allele_id_to_allele = reindexed_df[['index', 'allele']]
        reindexed_allele_df = reindexed_df.drop(['allele', 'index'], axis=1)
        return(guide_allele_id_to_allele, reindexed_allele_df)


    def transform_allele(self, adata, reindexed_df):
        '''
        Transform reindexed allele dataframe reindexed_df of (guide, allele_id_for_guide) -> (per sample count)
        to (n_reps, n_bins, n_guides, n_alleles) tensor.
        '''
        allele_tensor = torch.empty((self.n_reps, self.n_bins, self.n_guides, self.n_max_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            for j in range(self.n_bins):
                condit_idx = np.where((adata.condit.rep_id == i) & (adata.condit.sort_id == j))[0]
                assert len(condit_idx) == 1, print(i, j, condit_idx)
                condit_idx = condit_idx.item()
                condit_name = adata.condit.index[condit_idx]
                condit_allele_df = reindexed_df.loc[:,condit_name].unstack(level=1, fill_value = 0).astype(int)
                condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
                condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
                try:
                    assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, .".format(i, j)
                except Exception as e:
                    print(e)
                    return((condit_idx, condit_allele_df, condit_bcmatch_counts))
                condit_allele_df[0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
                condit_allele_df = condit_allele_df.sort_values("allele_id_for_guide", axis=1, ascending=True)
                assert all(condit_allele_df.columns == list(range(self.n_max_alleles)))
                assert condit_allele_df.to_numpy().shape == (self.n_guides, self.n_max_alleles)
                try:
                    allele_tensor[i, j, :, :] = torch.from_numpy(condit_allele_df.to_numpy())
                except:
                    return(allele_tensor, torch.from_numpy(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, reindexed_df)
        return(allele_tensor)


    def transform_allele_bulk(self, adata, reindexed_df):
        '''
        Transform reindexed allele dataframe reindexed_df of (guide, allele_id_for_guide) -> (per sample count)
        to (n_reps, n_bins, n_guides, n_alleles) tensor.
        '''
        
        allele_tensor = torch.empty((self.n_reps, 1, self.n_guides, self.n_max_alleles),  )
        if not self.device is None:
            allele_tensor = allele_tensor.cuda()
        for i in range(self.n_reps):
            condit_idx = np.where(adata.condit.rep_id == i)[0]
            assert len(condit_idx) == 1, print(i, "bulk", condit_idx)
            condit_idx = condit_idx.item()
            condit_name = adata.condit.index[condit_idx]
            condit_allele_df = reindexed_df.loc[:,condit_name].unstack(level=1, fill_value = 0).astype(int)
            condit_allele_df = condit_allele_df.reindex(adata.guides.index, fill_value = 0)
            condit_bcmatch_counts = adata.layers["X_bcmatch"][:, condit_idx].astype(int)
            try:
                assert (condit_bcmatch_counts >= condit_allele_df.sum(axis=1)).all(), "Allele counts are larger than total bcmatch counts in rep {}, bin {}.".format(i,)
            except Exception as e:
                print(e)
                return((condit_idx, condit_allele_df, condit_bcmatch_counts))
            condit_allele_df[0] = condit_bcmatch_counts - condit_allele_df.loc[:,condit_allele_df.columns != 0].sum(axis=1)
            condit_allele_df = condit_allele_df.sort_values("allele_id_for_guide", axis=1, ascending=True)
            assert all(condit_allele_df.columns == list(range(self.n_max_alleles)))
            assert condit_allele_df.to_numpy().shape == (self.n_guides, self.n_max_alleles)
            try:
                allele_tensor[i, 0, :, :] = torch.from_numpy(condit_allele_df.to_numpy())
            except:
                return(allele_tensor, torch.from_numpy(condit_allele_df.to_numpy()))
        try:
            assert (allele_tensor >= 0).all(), allele_tensor[allele_tensor < 0]
        except:
            return(allele_tensor, reindexed_df)
        return(allele_tensor)


    def get_allele_mask(self, adata, guide_allele_id_to_allele_df) -> torch.Tensor:
        """
        Returns allele mask of shape (n_guides, n_max_alleles) with boolean values
        (1 if valid, 0 if invalid allele)
        """
        guide_to_allele_tbl = guide_allele_id_to_allele_df.reset_index()
        n_valid_allele = guide_to_allele_tbl.groupby('guide')['allele_id_for_guide'].max().reindex(adata.guides.index, fill_value=0)
        #n_valid_allele[n_valid_allele == 0] = 1 # why do you want this?
        mask = torch.zeros((self.n_guides, self.n_max_alleles))
        for i in range(mask.shape[0]):
            mask[i, 0] = 1
            for j in range(n_valid_allele[i]):
                mask[i, j+1] = 1
        return mask.bool()

    def get_edit_to_index_dict(self, guide_allele_id_to_allele_df):
        '''
        Returns Dict[edit(str) -> index(int)] for all edits
        '''
        edit_lists = guide_allele_id_to_allele_df.allele.map(lambda a: list(a.edits))
        
        
        edits = pd.Series(pd.Series([e.get_abs_edit() for l in edit_lists.tolist() for e in l]).unique())
        edits_to_index = pd.Series(edits)
        edits_to_index = edits_to_index.reset_index().rename(columns={0:'edit'}).set_index('edit')['index'].to_dict()
        return(edits_to_index)

    def get_allele_to_edit_tensor(self, adata, edits_to_index: Dict[str, int], guide_allele_id_to_allele_df: pd.DataFrame):
        guide_allele_id_to_allele_df['edits'] = guide_allele_id_to_allele_df.allele.map(lambda a: list(a.edits))
        guide_allele_id_to_allele_df = guide_allele_id_to_allele_df.reset_index()
        guide_allele_id_to_allele_df['edit_idx'] = guide_allele_id_to_allele_df.edits.map(lambda es: [edits_to_index[e.get_abs_edit()] for e in es])
        guide_allele_id_to_edit_df = guide_allele_id_to_allele_df[['guide', 'allele_id_for_guide', 'edit_idx']].set_index(['guide', 'allele_id_for_guide'])
        guide_allele_id_to_edit_df = guide_allele_id_to_edit_df.unstack(level=1, fill_value=[]).reindex(
            adata.guides.index, fill_value=[])
        allele_edit_assignment = torch.zeros((len(adata.guides), self.n_max_alleles-1, len(edits_to_index.keys())))
        for i in range(len(guide_allele_id_to_edit_df)):
            for j in range(len(guide_allele_id_to_edit_df.columns)):
                allele_edit_assignment[i, j, guide_allele_id_to_edit_df.iloc[i,j]] = 1 
        return(allele_edit_assignment)


