import React from 'react';
import { CrowdData } from '../../types';
import { formatDate } from '../../utils/mock-data';

interface CrowdDensityChartProps {
  data: CrowdData[];
  location: string;
}

const CrowdDensityChart: React.FC<CrowdDensityChartProps> = ({ data, location }) => {
  // Find the maximum density for scaling
  const maxDensity = Math.max(...data.map(item => item.density));
  
  // Determine color based on current density (last data point)
  const currentDensity = data[data.length - 1].density;
  const getDensityColor = (density: number) => {
    if (density >= 80) return 'bg-red-500';
    if (density >= 60) return 'bg-orange-500';
    if (density >= 40) return 'bg-yellow-500';
    return 'bg-green-500';
  };
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <h2 className="font-semibold text-gray-900 dark:text-white">
            Crowd Density: {location}
          </h2>
          <div className="flex items-center">
            <div className={`h-2.5 w-2.5 rounded-full mr-1.5 ${getDensityColor(currentDensity)}`}></div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {currentDensity}% Capacity
            </span>
          </div>
        </div>
      </div>
      
      <div className="p-4 pt-0">
        <div className="mt-4 h-44 flex items-end space-x-2">
          {data.map((item, index) => (
            <div 
              key={index} 
              className="flex-1 flex flex-col items-center"
            >
              <div className="relative w-full h-full flex items-end justify-center">
                <div 
                  className={`w-full rounded-t transition-all duration-500 ${getDensityColor(item.density)}`}
                  style={{ 
                    height: `${(item.density / 100) * 100}%`,
                    opacity: index === data.length - 1 ? 1 : 0.7
                  }}
                />
                
                {/* Threshold line at 80% */}
                {index === 0 && (
                  <div className="absolute left-0 right-0 border-t-2 border-dashed border-red-400 dark:border-red-600" style={{ bottom: '80%' }}>
                    <span className="absolute right-full -top-3 text-xs text-red-500 pr-1">80%</span>
                  </div>
                )}
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {formatDate(item.timestamp)}
              </span>
            </div>
          ))}
        </div>
        
        <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center">
            <span className="inline-block h-3 w-3 bg-green-500 rounded-full mr-1"></span>
            <span>Low (&lt;40%)</span>
          </div>
          <div className="flex items-center">
            <span className="inline-block h-3 w-3 bg-yellow-500 rounded-full mr-1"></span>
            <span>Moderate (40-60%)</span>
          </div>
          <div className="flex items-center">
            <span className="inline-block h-3 w-3 bg-orange-500 rounded-full mr-1"></span>
            <span>High (60-80%)</span>
          </div>
          <div className="flex items-center">
            <span className="inline-block h-3 w-3 bg-red-500 rounded-full mr-1"></span>
            <span>Critical (&gt;80%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CrowdDensityChart;