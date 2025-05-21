import React from 'react';
import { 
  AlertCircle, ChevronRight, Shield, Users 
} from 'lucide-react';
import { Alert } from '../../types';
import { formatDate } from '../../utils/mock-data';

interface DetectionAlertsProps {
  alerts: Alert[];
}

const DetectionAlerts: React.FC<DetectionAlertsProps> = ({ alerts }) => {
  // Function to determine the icon based on alert type
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'weapon':
        return <Shield className="h-5 w-5 text-red-500" />;
      case 'suspicious_activity':
        return <AlertCircle className="h-5 w-5 text-amber-500" />;
      case 'crowd_density':
        return <Users className="h-5 w-5 text-blue-500" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />;
    }
  };
  
  // Function to determine the severity class
  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'high':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
        <h2 className="font-semibold text-gray-900 dark:text-white flex items-center">
          <AlertCircle className="h-5 w-5 mr-2 text-red-500" />
          Recent Alerts
        </h2>
        <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
          {alerts.length} incidents
        </span>
      </div>
      
      <div className="divide-y divide-gray-200 dark:divide-gray-700 max-h-96 overflow-y-auto">
        {alerts.map((alert) => (
          <div key={alert.id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
            <div className="flex items-start">
              <div className="mr-3 mt-0.5">
                {getAlertIcon(alert.type)}
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-start">
                  <h3 className="font-medium text-gray-900 dark:text-white text-sm">
                    {alert.description}
                  </h3>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full ml-2 ${getSeverityClass(alert.severity)}`}>
                    {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                  </span>
                </div>
                <div className="flex justify-between items-center mt-1.5">
                  <div className="flex items-center">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatDate(alert.timestamp)}
                    </span>
                    <span className="mx-1.5 text-gray-300 dark:text-gray-600">•</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      Camera: {alert.cameraId.replace('cam-', '')}
                    </span>
                  </div>
                  {!alert.acknowledged && (
                    <button className="text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors">
                      Acknowledge
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        <button className="w-full py-2 flex items-center justify-center text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">
          View all alerts
          <ChevronRight className="h-4 w-4 ml-1" />
        </button>
      </div>
    </div>
  );
};

export default DetectionAlerts;