import React from 'react';
import { Box, Paper, Typography, useTheme, LinearProgress, Button } from '@mui/material';
import { BarChart as BarChartIcon, ShowChart as LineChartIcon, PieChart as PieChartIcon } from '@mui/icons-material';

// Temporary chart component without Chart.js dependency
// This will be replaced with actual Chart.js implementation once dependencies are installed

interface ChartData {
  type: 'bar' | 'line' | 'pie' | 'doughnut';
  title: string;
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
    fill?: boolean;
  }>;
  chart_url?: string;
}

interface ChartComponentProps {
  chartData: ChartData;
}

const ChartComponent: React.FC<ChartComponentProps> = ({ chartData }) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  // Log chart data for debugging (can be removed in production)
  console.log('Chart visualization requested:', chartData?.title);

  // Generate Chart.js URL using QuickChart API
  const generateChartURL = () => {
    const config = {
      type: chartData.type,
      data: {
        labels: chartData.labels,
        datasets: chartData.datasets.map(dataset => ({
          ...dataset,
          backgroundColor: dataset.backgroundColor || [
            'rgba(99, 102, 241, 0.8)',
            'rgba(139, 92, 246, 0.8)',
            'rgba(6, 182, 212, 0.8)',
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)'
          ],
          borderColor: dataset.borderColor || [
            'rgb(99, 102, 241)',
            'rgb(139, 92, 246)',
            'rgb(6, 182, 212)',
            'rgb(16, 185, 129)',
            'rgb(245, 158, 11)',
            'rgb(239, 68, 68)'
          ],
          borderWidth: 2
        }))
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: chartData.title,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            display: true,
            position: 'top'
          }
        },
        scales: chartData.type !== 'pie' && chartData.type !== 'doughnut' ? {
          y: {
            beginAtZero: true
          }
        } : undefined
      }
    };

    const encodedConfig = encodeURIComponent(JSON.stringify(config));
    return `https://quickchart.io/chart?c=${encodedConfig}&width=600&height=400&format=png`;
  };

  // Add error handling for missing or invalid chart data
  if (!chartData || !chartData.datasets || !chartData.datasets[0] || !chartData.datasets[0].data) {
    return (
      <Paper sx={{
        p: 3,
        mt: 2,
        background: isDarkMode
          ? 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
          : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
        border: `1px solid ${isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'}`,
        borderRadius: 3,
        textAlign: 'center'
      }}>
        <Typography variant="h6" color="error.main" gutterBottom>
          Chart Data Error
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Unable to display chart: Invalid or missing chart data
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Chart Data: {JSON.stringify(chartData, null, 2)}
        </Typography>
      </Paper>
    );
  }

  // Get chart icon based on type
  const getChartIcon = () => {
    switch (chartData.type) {
      case 'bar':
        return <BarChartIcon sx={{ fontSize: 32, color: 'primary.main' }} />;
      case 'line':
        return <LineChartIcon sx={{ fontSize: 32, color: 'primary.main' }} />;
      case 'pie':
      case 'doughnut':
        return <PieChartIcon sx={{ fontSize: 32, color: 'primary.main' }} />;
      default:
        return <BarChartIcon sx={{ fontSize: 32, color: 'primary.main' }} />;
    }
  };

  // Calculate max value for progress bars (with safety check)
  const dataValues = chartData.datasets[0].data || [];
  const maxValue = dataValues.length > 0 ? Math.max(...dataValues) : 1;

  const renderChart = () => {
    return (
      <Box>
        {/* Chart Header */}
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          mb: 4,
          p: 2,
          bgcolor: 'primary.50',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'primary.200'
        }}>
          {getChartIcon()}
          <Box>
            <Typography variant="h5" fontWeight="bold" color="primary.main">
              {chartData.title}
            </Typography>
            <Typography variant="body1" color="primary.dark" fontWeight="500">
              {chartData.type.charAt(0).toUpperCase() + chartData.type.slice(1)} Chart Visualization
            </Typography>
          </Box>
        </Box>

        {/* Data Visualization */}
        <Box sx={{ mb: 4 }}>
          {(chartData.labels || []).map((label, index) => {
            const value = chartData.datasets[0].data[index] || 0;
            const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;

            return (
              <Box key={index} sx={{
                mb: 3,
                p: 2,
                bgcolor: 'action.hover',
                borderRadius: 2,
                border: '1px solid',
                borderColor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2" fontWeight="500">
                    {label}
                  </Typography>
                  <Typography variant="body2" fontWeight="bold" color="primary.main">
                    {value}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={percentage}
                  sx={{
                    height: 12,
                    borderRadius: 6,
                    bgcolor: isDarkMode ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)',
                    border: '1px solid',
                    borderColor: isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 6,
                      background: `linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)`,
                      boxShadow: '0 2px 8px rgba(99, 102, 241, 0.3)',
                    }
                  }}
                />
              </Box>
            );
          })}
        </Box>

        {/* Dataset Info */}
        <Box sx={{
          mt: 3,
          p: 3,
          bgcolor: 'success.50',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'success.200'
        }}>
          <Typography variant="subtitle2" color="success.main" fontWeight="bold" gutterBottom>
            📊 Dataset: {chartData.datasets[0].label}
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2, mt: 2 }}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" fontWeight="bold" color="success.main">
                {chartData.labels.length}
              </Typography>
              <Typography variant="caption" color="success.dark">
                Data Points
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" fontWeight="bold" color="success.main">
                {maxValue}
              </Typography>
              <Typography variant="caption" color="success.dark">
                Max Value
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" fontWeight="bold" color="success.main">
                {chartData.datasets[0].data.reduce((a, b) => a + b, 0)}
              </Typography>
              <Typography variant="caption" color="success.dark">
                Total Sum
              </Typography>
            </Box>
          </Box>
        </Box>



        {/* Interactive Chart Image */}
        <Box sx={{
          mt: 3,
          p: 3,
          bgcolor: 'success.50',
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'success.200',
          textAlign: 'center'
        }}>
          <Typography variant="h6" color="success.main" fontWeight="bold" gutterBottom>
            📊 Full Interactive Chart
          </Typography>

          {/* Chart Image */}
          <Box sx={{
            mt: 2,
            mb: 3,
            p: 2,
            bgcolor: 'white',
            borderRadius: 2,
            border: '2px solid',
            borderColor: 'success.300',
            display: 'inline-block'
          }}>
            <img
              src={chartData.chart_url || generateChartURL()}
              alt={`${chartData.title} - ${chartData.type} chart`}
              style={{
                maxWidth: '100%',
                height: 'auto',
                borderRadius: '8px'
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.nextElementSibling!.textContent = 'Chart image failed to load';
              }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'none' }}>
              Chart image failed to load
            </Typography>
          </Box>

          <Typography variant="body2" color="success.dark" sx={{ mb: 2 }}>
            Generated using Chart.js via QuickChart API
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              color="success"
              onClick={() => window.open(chartData.chart_url || generateChartURL(), '_blank')}
              sx={{ minWidth: 120 }}
            >
              Open Full Size
            </Button>
            <Button
              variant="outlined"
              color="success"
              onClick={() => {
                const url = chartData.chart_url || generateChartURL();
                navigator.clipboard.writeText(url);
              }}
              sx={{ minWidth: 120 }}
            >
              Copy Chart URL
            </Button>
          </Box>
        </Box>
      </Box>
    );
  };

  return (
    <Paper sx={{
      p: 3,
      mt: 2,
      background: isDarkMode
        ? 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
        : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      border: `1px solid ${isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'}`,
      borderRadius: 3,
      boxShadow: isDarkMode
        ? '0 8px 32px rgba(0, 0, 0, 0.3)'
        : '0 8px 32px rgba(0, 0, 0, 0.08)'
    }}>
      <Box sx={{ minHeight: 300, width: '100%', p: 2 }}>
        {renderChart()}
      </Box>
    </Paper>
  );
};

export default ChartComponent;