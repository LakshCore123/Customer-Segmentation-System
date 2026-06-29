import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import logging
import os
from datetime import datetime
import plotly.express as px

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CustomerSegmentation:
    """RFM + KMeans customer segmentation pipeline."""

    def __init__(self, data_path='online_retail.csv'):
        self.data_path = data_path
        self.data = None
        self.rfm_df = None
        self.scaled_features = None
        self.scaler = None
        self.kmeans = None
        self.clusters = None
        self.optimal_k = None
        self.segment_profiles = None

        self.output_dir = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(f'{self.output_dir}/figures', exist_ok=True)
        logger.info(f"Output dir: {self.output_dir}")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path, encoding="utf-8-sig")
            # csv sometimes comes in with a BOM / stray whitespace in headers
            self.data.columns = self.data.columns.str.replace('\ufeff', '', regex=False).str.strip()
            logger.info(f"Loaded {self.data_path}, shape={self.data.shape}")
        except FileNotFoundError:
            logger.warning(f"{self.data_path} not found, generating dummy data instead")
            n = 50000
            self.data = pd.DataFrame({
                'InvoiceNo': np.random.randint(10000, 99999, n).astype(str),
                'StockCode': np.random.choice(['A001', 'B002', 'C003', 'D004'], n),
                'Description': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n),
                'Quantity': np.random.choice(range(1, 20), n),
                'InvoiceDate': pd.date_range('2023-01-01', '2023-12-31', periods=n),
                'UnitPrice': np.random.uniform(1, 50, n),
                'CustomerID': np.random.randint(10000, 15000, n),
                'Country': np.random.choice(['UK', 'USA', 'Germany', 'France', 'Australia'], n)
            })
            self.data['TotalPrice'] = self.data['Quantity'] * self.data['UnitPrice']

        return self.data

    def data_audit(self):
        logger.info(f"shape={self.data.shape}, duplicates={self.data.duplicated().sum()}")
        logger.info(f"missing values:\n{self.data.isnull().sum()}")

        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/missing_values.png', dpi=300)
        plt.close()

    def clean_data(self):
        before = self.data.shape[0]

        self.data = self.data.dropna(subset=['CustomerID'])
        self.data = self.data.drop_duplicates()
        self.data = self.data[~self.data['InvoiceNo'].astype(str).str.contains('C', na=False)]  # cancellations
        self.data = self.data[(self.data['Quantity'] > 0) & (self.data['UnitPrice'] > 0)]

        self.data['TotalPrice'] = self.data['Quantity'] * self.data['UnitPrice']
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'], format="%d-%m-%Y %H:%M", errors="coerce")
        self.data = self.data.dropna(subset=['InvoiceDate'])
        self.data['YearMonth'] = self.data['InvoiceDate'].dt.strftime('%Y-%m')

        logger.info(f"cleaning dropped {before - self.data.shape[0]} rows")
        return self.data

    def eda(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.data['Quantity'].hist(bins=50, ax=axes[0, 0])
        axes[0, 0].set_title('Quantity')
        self.data['UnitPrice'].hist(bins=50, ax=axes[0, 1])
        axes[0, 1].set_title('Unit Price')
        self.data['TotalPrice'].hist(bins=50, ax=axes[1, 0])
        axes[1, 0].set_title('Total Price')
        self.data['Country'].value_counts().head(10).plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Countries')
        axes[1, 1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/eda_distributions.png', dpi=300)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['Quantity', 'UnitPrice', 'TotalPrice']):
            self.data.boxplot(column=col, ax=axes[i])
            axes[i].set_title(col)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/eda_boxplots.png', dpi=300)
        plt.close()

    def calculate_rfm(self):
        ref_date = self.data['InvoiceDate'].max() + pd.Timedelta(days=1)

        rfm = self.data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()

        self.rfm_df = rfm
        logger.info(f"RFM computed for {len(rfm)} customers")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
            rfm[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(col)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/rfm_distributions.png', dpi=300)
        plt.close()

        return rfm

    def prepare_features(self):
        features = self.rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

        # cap outliers with IQR rather than dropping them - don't want to lose customers
        for col in features.columns:
            q1, q3 = features[col].quantile(0.25), features[col].quantile(0.75)
            iqr = q3 - q1
            features[col] = features[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        features['Frequency'] = np.log1p(features['Frequency'])
        features['Monetary'] = np.log1p(features['Monetary'])

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(features)

        plt.figure(figsize=(8, 6))
        sns.heatmap(features.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('RFM Correlation')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/rfm_correlation.png', dpi=300)
        plt.close()

        return self.scaled_features

    def run_kmeans(self, k_range=(2, 10)):
        results = {}
        inertias, sil_scores = [], []

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
            labels = km.fit_predict(self.scaled_features)
            sil = silhouette_score(self.scaled_features, labels)

            results[k] = {'model': km, 'labels': labels, 'inertia': km.inertia_, 'silhouette': sil}
            inertias.append(km.inertia_)
            sil_scores.append(sil)
            logger.info(f"k={k} inertia={km.inertia_:.1f} silhouette={sil:.4f}")

        ks = range(k_range[0], k_range[1] + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(ks, inertias, 'bo-')
        plt.xlabel('K')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.xticks(ks)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/elbow_curve.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(ks, sil_scores, 'ro-')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.axhline(y=0.35, color='green', linestyle='--', label='target (0.35)')
        plt.xticks(ks)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/silhouette_analysis.png', dpi=300)
        plt.close()

        return results

    def pick_best_k(self, results):
        best_k = max(results, key=lambda k: results[k]['silhouette'])

        if results[best_k]['silhouette'] < 0.35:
            logger.warning(f"best silhouette {results[best_k]['silhouette']:.4f} below 0.35, looking for alternative")
            for k, r in results.items():
                if r['silhouette'] >= 0.35:
                    best_k = k
                    break

        self.optimal_k = best_k
        self.kmeans = results[best_k]['model']
        self.clusters = results[best_k]['labels']
        logger.info(f"using k={best_k}, silhouette={results[best_k]['silhouette']:.4f}")
        return best_k

    def pca_plot(self):
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_result = pca.fit_transform(self.scaled_features)

        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': self.clusters.astype(str),
            'Recency': self.rfm_df['Recency'].values,
            'Frequency': self.rfm_df['Frequency'].values,
            'Monetary': self.rfm_df['Monetary'].values
        })

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=self.clusters, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter)
        plt.title('Customer Segments (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/pca_visualization.png', dpi=300)
        plt.close()

        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Cluster',
            hover_data=['Recency', 'Frequency', 'Monetary'],
            title='Customer Segments (interactive)'
        )
        fig.write_html(f'{self.output_dir}/figures/pca_interactive.html')

    def build_profiles(self):
        rfm = self.rfm_df.copy()
        rfm['Cluster'] = self.clusters

        rows = []
        for c in sorted(rfm['Cluster'].unique()):
            sub = rfm[rfm['Cluster'] == c]
            rows.append({
                'Cluster': c,
                'Customer_Count': len(sub),
                'Percentage': len(sub) / len(rfm) * 100,
                'Avg_Recency': sub['Recency'].mean(),
                'Avg_Frequency': sub['Frequency'].mean(),
                'Avg_Monetary': sub['Monetary'].mean(),
                'Median_Recency': sub['Recency'].median(),
                'Median_Frequency': sub['Frequency'].median(),
                'Median_Monetary': sub['Monetary'].median(),
            })

        profiles = pd.DataFrame(rows).sort_values('Avg_Monetary', ascending=False)

        # rough rule-based naming off the RFM quantiles
        r_q25, r_q75 = rfm['Recency'].quantile([0.25, 0.75])
        f_q50, f_q75 = rfm['Frequency'].quantile([0.5, 0.75])
        m_q50, m_q75 = rfm['Monetary'].quantile([0.5, 0.75])

        names = []
        for _, row in profiles.iterrows():
            if row['Avg_Monetary'] > m_q75:
                if row['Avg_Recency'] < r_q25:
                    names.append('VIP Customers')
                elif row['Avg_Frequency'] > f_q75:
                    names.append('Loyal Customers')
                else:
                    names.append('Potential Loyalists')
            elif row['Avg_Recency'] > r_q75:
                names.append('At Risk' if row['Avg_Monetary'] > m_q50 else 'Lost Customers')
            else:
                names.append('Active Customers' if row['Avg_Frequency'] > f_q50 else 'New Customers')

        profiles['Segment_Name'] = names
        self.segment_profiles = profiles
        logger.info(f"\n{profiles[['Cluster', 'Segment_Name', 'Customer_Count', 'Avg_Monetary']]}")
        return profiles

    def marketing_strategies(self):
        rows = []
        for _, row in self.segment_profiles.iterrows():
            seg = row['Segment_Name']
            rows.append({
                'Segment': seg,
                'Customer_Count': row['Customer_Count'],
                'Avg_Monetary': row['Avg_Monetary'],
                'Avg_Recency': row['Avg_Recency'],
                'Avg_Frequency': row['Avg_Frequency'],
                'Discount_Strategy': self._discount(seg),
                'Email_Campaign': self._email(seg),
                'Loyalty_Program': self._loyalty(seg),
                'Cross_Selling': self._cross_sell(row['Avg_Monetary']),
                'Retention_Strategy': self._retention(seg),
                'Reengagement_Strategy': self._reengagement(row['Avg_Recency']),
            })

        df = pd.DataFrame(rows)
        df.to_csv(f'{self.output_dir}/marketing_strategies.csv', index=False)
        return df

    @staticmethod
    def _discount(seg):
        if 'VIP' in seg:
            return "Premium loyalty discounts (15-20%) + early access to new collections"
        if 'Loyal' in seg:
            return "Tiered discounts based on purchase frequency (10-15%)"
        if 'Potential' in seg:
            return "First-time buyer incentives (20% off) + bundle discounts"
        if 'At Risk' in seg:
            return "Win-back discounts (20-25%) + personalized offers"
        if 'Lost' in seg:
            return "Aggressive re-engagement discounts (30-40%) + free shipping"
        return "Standard promotional offers (10-15%)"

    @staticmethod
    def _email(seg):
        if 'VIP' in seg:
            return "Monthly VIP newsletters + exclusive event invitations"
        if 'Loyal' in seg:
            return "Weekly personalized recommendations + loyalty updates"
        if 'Potential' in seg:
            return "Welcome series (5 emails) + product education content"
        if 'At Risk' in seg:
            return "Re-engagement drip campaign + personalized win-back offers"
        if 'Lost' in seg:
            return "Re-engagement campaigns with surveys + exclusive deals"
        return "Regular promotional emails (2-3 per week)"

    @staticmethod
    def _loyalty(seg):
        if 'VIP' in seg:
            return "Tier 1: 5x points + free shipping + exclusive events"
        if 'Loyal' in seg:
            return "Tier 2: 3x points + early access to sales"
        if 'Potential' in seg:
            return "Welcome bonus (500 points) + first purchase multiplier"
        if 'At Risk' in seg:
            return "Bonus points for reactivation + double points on next purchase"
        if 'Lost' in seg:
            return "Return customer bonus + loyalty re-welcome gift"
        return "Standard loyalty program (1x points per purchase)"

    @staticmethod
    def _cross_sell(monetary):
        if monetary > 5000:
            return "Premium product recommendations + complementary accessories"
        if monetary > 2000:
            return "Personalized product bundles + 'frequently bought together' suggestions"
        if monetary > 500:
            return "Up-selling opportunities + product category expansions"
        return "Introductory products + starter bundles"

    @staticmethod
    def _retention(seg):
        if 'VIP' in seg:
            return "Dedicated account manager + priority support + birthday gifts"
        if 'Loyal' in seg:
            return "Regular engagement + loyalty milestone rewards"
        if 'Potential' in seg:
            return "Onboarding sequence + personalized shopping guides"
        if 'At Risk' in seg:
            return "Warning notifications + personalized incentives"
        if 'Lost' in seg:
            return "Win-back programs + re-engagement analysis"
        return "Standard retention communications"

    @staticmethod
    def _reengagement(recency):
        if recency > 90:
            return "Win-back email series + reactivation offers + customer feedback survey"
        if recency > 60:
            return "Re-engagement campaigns + 'we miss you' emails + special offers"
        if recency > 30:
            return "Engagement reminders + new product alerts"
        return "Regular engagement + satisfaction surveys"

    def segment_pie_chart(self):
        plt.figure(figsize=(10, 8))
        counts = self.segment_profiles.set_index('Segment_Name')['Customer_Count']
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Customer Segment Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/segment_distribution.png', dpi=300)
        plt.close()

    def write_report(self, strategies):
        lines = []
        lines.append("CUSTOMER SEGMENTATION - FINAL REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total customers: {len(self.rfm_df):,}")
        lines.append(f"Number of segments: {self.optimal_k}")
        lines.append(f"Silhouette score: {silhouette_score(self.scaled_features, self.clusters):.4f}")

        lines.append("\nSEGMENT SUMMARY")
        for _, row in self.segment_profiles.iterrows():
            lines.append(f"\n{row['Segment_Name']} (cluster {row['Cluster']})")
            lines.append(f"  customers: {row['Customer_Count']:,} ({row['Percentage']:.1f}%)")
            lines.append(f"  avg monetary: ${row['Avg_Monetary']:,.2f}")
            lines.append(f"  avg recency: {row['Avg_Recency']:.1f} days")
            lines.append(f"  avg frequency: {row['Avg_Frequency']:.1f}")

        lines.append("\nMARKETING RECOMMENDATIONS")
        for _, row in strategies.iterrows():
            lines.append(f"\n{row['Segment']}")
            lines.append(f"  discounts: {row['Discount_Strategy']}")
            lines.append(f"  email: {row['Email_Campaign']}")
            lines.append(f"  loyalty: {row['Loyalty_Program']}")
            lines.append(f"  cross-sell: {row['Cross_Selling']}")
            lines.append(f"  retention: {row['Retention_Strategy']}")
            lines.append(f"  re-engagement: {row['Reengagement_Strategy']}")

        text = "\n".join(lines)
        with open(f'{self.output_dir}/final_report.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        with open(f'{self.output_dir}/final_report.md', 'w', encoding='utf-8') as f:
            f.write(text)

        logger.info(f"report written to {self.output_dir}")

    def run(self):
        start = datetime.now()

        self.load_data()
        self.data_audit()
        self.clean_data()
        self.eda()

        self.calculate_rfm()
        self.prepare_features()

        kmeans_results = self.run_kmeans(k_range=(2, 10))
        self.pick_best_k(kmeans_results)

        self.pca_plot()
        self.build_profiles()
        self.segment_pie_chart()

        strategies = self.marketing_strategies()
        self.write_report(strategies)

        logger.info(f"pipeline finished in {datetime.now() - start}")

        return {
            'data_shape': self.data.shape,
            'n_customers': len(self.rfm_df),
            'optimal_k': self.optimal_k,
            'silhouette_score': silhouette_score(self.scaled_features, self.clusters),
            'segment_profiles': self.segment_profiles,
            'marketing_strategies': strategies,
            'output_directory': self.output_dir
        }


def main():
    system = CustomerSegmentation(data_path='online_retail.csv')
    results = system.run()

    print("\nCUSTOMER SEGMENTATION - SUMMARY")
    print(f"Total customers: {results['n_customers']:,}")
    print(f"Segments: {results['optimal_k']}")
    print(f"Silhouette score: {results['silhouette_score']:.4f}")
    print(f"Output dir: {results['output_directory']}")
    print("\nSegment profiles:")
    print(results['segment_profiles'][['Segment_Name', 'Customer_Count', 'Avg_Monetary', 'Avg_Recency']].to_string(index=False))

    return results


if __name__ == "__main__":
    main()
