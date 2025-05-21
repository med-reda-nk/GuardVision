import React from 'react';
import { ArrowDown, ArrowUp, Bell, ShieldAlert, Eye, Clock } from 'lucide-react';
import { SecurityStat } from '../../types';

interface StatisticsCardProps {
  stat: SecurityStat;
}

const StatisticsCard: React.FC<StatisticsCardProps> = ({ stat }) => {
  const getIcon = () => {
    switch (stat.icon) {
      case 'Bell':
        return <Bell className="h-5 w-5 text-blue-500" />;
      case 'ShieldAlert':
        return <ShieldAlert className="h-5 w-5 text-blue-500" />;
      case 'Eye':
        return <Eye className="h-5 w-5 text-blue-500" />;
      case 'Clock':
        return <Clock className="h-5 w-5 text-blue-500" />;
      default:
        console.warn(`Icon not mapped: ${stat.icon}`);
        return null;
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between">
        <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
          {getIcon()}
        </div>
        
        <div className={`flex items-center text-sm font-medium ${
          stat.change > 0 ? 'text-red-500' : 'text-green-500'
        }`}>
          {stat.change > 0 ? (
            <>
              <ArrowUp className="h-3 w-3 mr-1" />
              {Math.abs(stat.change)}%
            </>
          ) : (
            <>
              <ArrowDown className="h-3 w-3 mr-1" />
              {Math.abs(stat.change)}%
            </>
          )}
        </div>
      </div>
      
      <div className="mt-3">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
          {typeof stat.value === 'number' && Number.isInteger(stat.value) ? stat.value : stat.value.toFixed(1)}
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{stat.label}</p>
      </div>
    </div>
  );
};

export default StatisticsCard;