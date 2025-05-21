import React, { useState } from 'react';
import { Bell, Menu, Moon, Sun, User, X } from 'lucide-react';
import { mockAlerts } from '../../utils/mock-data';

const Header: React.FC<{ toggleSidebar: () => void }> = ({ toggleSidebar }) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  
  const unacknowledgedAlerts = mockAlerts.filter(alert => !alert.acknowledged);
  
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };
  
  return (
    <header className="flex items-center justify-between px-4 md:px-6 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 z-30">
      <div className="flex items-center space-x-4">
        <button 
          onClick={toggleSidebar} 
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Toggle menu"
        >
          <Menu className="h-5 w-5 text-gray-600 dark:text-gray-300" />
        </button>
        <div className="flex items-center">
          <h1 className="text-lg font-bold text-gray-900 dark:text-white">SecureVision</h1>
          <span className="ml-2 bg-blue-600 text-white px-2 py-0.5 rounded-md text-xs font-medium">BETA</span>
        </div>
      </div>
      
      <div className="flex items-center space-x-3">
        <button 
          onClick={toggleDarkMode} 
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Toggle dark mode"
        >
          {isDarkMode ? 
            <Sun className="h-5 w-5 text-gray-600 dark:text-gray-300" /> : 
            <Moon className="h-5 w-5 text-gray-600 dark:text-gray-300" />
          }
        </button>
        
        <div className="relative">
          <button 
            onClick={() => setShowNotifications(!showNotifications)} 
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors relative"
            aria-label="Notifications"
          >
            <Bell className="h-5 w-5 text-gray-600 dark:text-gray-300" />
            {unacknowledgedAlerts.length > 0 && (
              <span className="absolute top-0 right-0 h-4 w-4 bg-red-500 rounded-full flex items-center justify-center text-white text-xs">
                {unacknowledgedAlerts.length}
              </span>
            )}
          </button>
          
          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50 overflow-hidden">
              <div className="p-3 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                <h3 className="font-semibold text-gray-900 dark:text-white">Notifications</h3>
                <button 
                  onClick={() => setShowNotifications(false)}
                  className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="max-h-96 overflow-y-auto">
                {unacknowledgedAlerts.length === 0 ? (
                  <p className="p-4 text-gray-500 dark:text-gray-400 text-center">No new notifications</p>
                ) : (
                  unacknowledgedAlerts.map(alert => (
                    <div key={alert.id} className="p-3 border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                      <div className="flex items-start">
                        <div className={`h-2 w-2 mt-1.5 rounded-full mr-2 ${
                          alert.severity === 'critical' ? 'bg-red-500' :
                          alert.severity === 'high' ? 'bg-orange-500' :
                          alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                        }`} />
                        <div>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">{alert.description}</p>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            {alert.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              {unacknowledgedAlerts.length > 0 && (
                <div className="p-3 border-t border-gray-200 dark:border-gray-700">
                  <button className="w-full py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors">
                    Mark all as read
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="border-l border-gray-300 dark:border-gray-700 h-6 mx-1"></div>
        
        <button className="flex items-center space-x-2 py-1 px-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
          <div className="relative h-8 w-8 rounded-full bg-gray-300 dark:bg-gray-700 overflow-hidden flex items-center justify-center">
            <User className="h-5 w-5 text-gray-600 dark:text-gray-400" />
          </div>
          <span className="text-sm font-medium text-gray-800 dark:text-gray-200 hidden sm:inline">
            Security Admin
          </span>
        </button>
      </div>
    </header>
  );
};

export default Header;