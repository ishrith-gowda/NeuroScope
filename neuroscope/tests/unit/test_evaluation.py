"""
evaluation metrics tests.

unit tests for quality metrics and statistical analysis.
"""

import pytest
import torch
import numpy as np


class TestSSIM:
    """test ssim metric."""
    
    def test_ssim_identical_images(self):
        """test ssim of identical images is 1."""
        from ..evaluation.metrics import SSIMMetric
        
        metric = SSIMMetric()
        
        x = torch.randn(2, 1, 64, 64, 64)
        ssim = metric(x, x)
        
        assert abs(ssim - 1.0) < 0.01
    
    def test_ssim_different_images(self):
        """test ssim of different images is less than 1."""
        from ..evaluation.metrics import SSIMMetric
        
        metric = SSIMMetric()
        
        x = torch.randn(2, 1, 64, 64, 64)
        y = torch.randn(2, 1, 64, 64, 64)
        
        ssim = metric(x, y)
        
        assert ssim < 1.0
        assert ssim > -1.0
    
    def test_ssim_range(self):
        """test ssim is in valid range."""
        from ..evaluation.metrics import SSIMMetric
        
        metric = SSIMMetric()
        
        for _ in range(10):
            x = torch.randn(1, 1, 32, 32, 32)
            y = torch.randn(1, 1, 32, 32, 32)
            
            ssim = metric(x, y)
            
            assert -1.0 <= ssim <= 1.0


class TestPSNR:
    """test psnr metric."""
    
    def test_psnr_identical_images(self):
        """test psnr of identical images is very high."""
        from ..evaluation.metrics import PSNRMetric
        
        metric = PSNRMetric()
        
        x = torch.randn(2, 1, 64, 64, 64)
        psnr = metric(x, x)
        
        # identical images should have very high psnr
        assert psnr > 50.0
    
    def test_psnr_different_images(self):
        """test psnr of different images."""
        from ..evaluation.metrics import PSNRMetric
        
        metric = PSNRMetric()
        
        x = torch.randn(2, 1, 64, 64, 64)
        y = x + 0.1 * torch.randn_like(x)  # add noise
        
        psnr = metric(x, y)
        
        # should be positive but not infinite
        assert 10.0 < psnr < 50.0
    
    def test_psnr_symmetry(self):
        """test psnr is symmetric."""
        from ..evaluation.metrics import PSNRMetric
        
        metric = PSNRMetric()
        
        x = torch.randn(2, 1, 32, 32, 32)
        y = torch.randn(2, 1, 32, 32, 32)
        
        psnr_xy = metric(x, y)
        psnr_yx = metric(y, x)
        
        assert abs(psnr_xy - psnr_yx) < 0.01


class TestFID:
    """test fid metric."""
    
    @pytest.mark.slow
    def test_fid_same_distribution(self):
        """test fid of same distribution is low."""
        from ..evaluation.metrics import FIDMetric
        
        metric = FIDMetric()
        
        # same distribution should have low fid
        real = torch.randn(100, 512)
        fake = torch.randn(100, 512)
        
        fid = metric.compute_from_features(real, fake)
        
        assert fid >= 0
    
    def test_fid_different_distributions(self):
        """test fid of different distributions."""
        from ..evaluation.metrics import FIDMetric
        
        metric = FIDMetric()
        
        # different distributions should have higher fid
        real = torch.randn(100, 512)
        fake = torch.randn(100, 512) + 2.0  # shifted distribution
        
        fid_same = metric.compute_from_features(real, real)
        fid_diff = metric.compute_from_features(real, fake)
        
        assert fid_diff > fid_same


class TestLPIPS:
    """test lpips metric."""
    
    def test_lpips_identical_images(self):
        """test lpips of identical images is 0."""
        from ..evaluation.metrics import LPIPSMetric
        
        metric = LPIPSMetric()
        
        x = torch.randn(2, 3, 64, 64)
        lpips = metric(x, x)
        
        assert lpips < 0.01
    
    def test_lpips_range(self):
        """test lpips is non-negative."""
        from ..evaluation.metrics import LPIPSMetric
        
        metric = LPIPSMetric()
        
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        
        lpips = metric(x, y)
        
        assert lpips >= 0


class TestStatisticalTests:
    """test statistical analysis functions."""
    
    def test_wilcoxon_test(self):
        """test wilcoxon signed-rank test."""
        from ..evaluation.statistical import StatisticalTester
        
        tester = StatisticalTester()
        
        # create clearly different samples
        x = np.random.randn(30) + 2.0
        y = np.random.randn(30)
        
        result = tester.wilcoxon_test(x, y)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['p_value'] < 0.05  # should be significant
    
    def test_mann_whitney_test(self):
        """test mann-whitney u test."""
        from ..evaluation.statistical import StatisticalTester
        
        tester = StatisticalTester()
        
        x = np.random.randn(30) + 2.0
        y = np.random.randn(30)
        
        result = tester.mann_whitney_test(x, y)
        
        assert 'statistic' in result
        assert 'p_value' in result
    
    def test_effect_size(self):
        """test cohen's d effect size."""
        from ..evaluation.statistical import EffectSizeCalculator
        
        calc = EffectSizeCalculator()
        
        x = np.random.randn(100)
        y = np.random.randn(100) + 1.0  # 1 sd difference
        
        d = calc.cohens_d(x, y)
        
        # should be around 1.0 (large effect)
        assert 0.5 < abs(d) < 1.5
    
    def test_bonferroni_correction(self):
        """test bonferroni correction."""
        from ..evaluation.statistical import bonferroni_correction
        
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected = bonferroni_correction(p_values)
        
        # corrected p-values should be higher
        for orig, corr in zip(p_values, corrected):
            assert corr >= orig


class TestMetricAggregation:
    """test metric aggregation."""
    
    def test_metric_calculator(self):
        """test metriccalculator aggregation."""
        from ..evaluation.metrics import MetricCalculator
        
        calc = MetricCalculator()
        
        pred = torch.randn(5, 1, 32, 32, 32)
        target = torch.randn(5, 1, 32, 32, 32)
        
        metrics = calc.compute_all(pred, target)
        
        assert 'ssim' in metrics
        assert 'psnr' in metrics
    
    def test_regional_analysis(self):
        """test regional metric analysis."""
        from ..evaluation.analyzers import RegionalAnalyzer
        
        analyzer = RegionalAnalyzer()
        
        pred = np.random.randn(64, 64, 64)
        target = np.random.randn(64, 64, 64)
        mask = np.zeros((64, 64, 64))
        mask[20:40, 20:40, 20:40] = 1  # define region
        
        result = analyzer.analyze_region(pred, target, mask)
        
        assert 'ssim' in result
        assert 'mse' in result


class TestReporters:
    """test report generation."""
    
    def test_latex_reporter(self, tmp_path):
        """test latex report generation."""
        from ..evaluation.reporters import LaTeXReporter
        
        reporter = LaTeXReporter()
        
        results = {
            'method1': {'ssim': 0.90, 'psnr': 28.5},
            'method2': {'ssim': 0.92, 'psnr': 29.1}
        }
        
        table = reporter.create_table(results, ['ssim', 'psnr'])
        
        assert '\\begin{table}' in table
        assert 'method1' in table or 'Method1' in table
    
    def test_markdown_reporter(self, tmp_path):
        """test markdown report generation."""
        from ..evaluation.reporters import MarkdownReporter
        
        reporter = MarkdownReporter()
        
        results = {
            'method1': {'ssim': 0.90, 'psnr': 28.5},
            'method2': {'ssim': 0.92, 'psnr': 29.1}
        }
        
        table = reporter.create_table(results, ['ssim', 'psnr'])
        
        assert '|' in table
        assert 'ssim' in table.lower() or 'SSIM' in table
    
    def test_json_export(self, tmp_path):
        """test json export."""
        from ..evaluation.reporters import JSONReporter
        import json
        
        reporter = JSONReporter()
        
        results = {
            'ssim': 0.92,
            'psnr': 29.5,
            'fid': 45.2
        }
        
        output_path = tmp_path / 'results.json'
        reporter.save(results, str(output_path))
        
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['ssim'] == 0.92
