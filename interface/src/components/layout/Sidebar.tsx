import React from 'react';
import { 
  BarChart3, Camera, Clock, Cog, Home, 
  Map, ShieldAlert, Users, X 
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const navItems = [
    { label: 'Dashboard', icon: <Home />, active: true },
    { label: 'Live Monitoring', icon: <Camera /> },
    { label: 'Event Timeline', icon: <Clock /> },
    { label: 'Threat Detection', icon: <ShieldAlert /> },
    { label: 'Analytics', icon: <BarChart3 /> },
    { label: 'Crowd Monitoring', icon: <Users /> },
    { label: 'Site Map', icon: <Map /> },
    { label: 'Settings', icon: <Cog /> },
  ];

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={onClose}
        />
      )}
      
      <aside 
        className={`fixed top-0 left-0 h-full w-64 bg-gray-900 text-white shadow-lg transform transition-transform duration-300 ease-in-out z-50 ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div className="p-4 flex items-center justify-between">
          <div className="flex items-center">
            <ShieldAlert className="h-6 w-6 text-blue-400 mr-2" />
            <h2 className="text-xl font-bold">SecureVision</h2>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-800 md:hidden"
            aria-label="Close menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <div className="px-2">
          <div className="h-px bg-gray-700 my-3" />
          
          <nav className="mt-4 space-y-1">
            {navItems.map((item, index) => (
              <a
                key={index}
                href="#"
                className={`flex items-center px-3 py-2.5 rounded-lg ${
                  item.active 
                    ? 'bg-blue-700 text-white' 
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                } transition-colors group`}
              >
                <span className={`mr-3 ${item.active ? 'text-white' : 'text-gray-400 group-hover:text-white'}`}>
                  {React.cloneElement(item.icon, { className: 'h-5 w-5' })}
                </span>
                <span className="text-sm font-medium">{item.label}</span>
                {item.label === 'Threat Detection' && (
                  <span className="ml-auto bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full">3</span>
                )}
              </a>
            ))}
          </nav>
        </div>
        
        <div className="absolute bottom-0 left-0 right-0 p-4">
          <div className="bg-gray-800 rounded-lg p-3 text-center">
            <p className="text-sm text-gray-400 mb-1">System Status</p>
            <p className="text-sm text-white flex justify-center items-center">
              <span className="h-2 w-2 bg-green-500 rounded-full mr-1.5"></span>
              All Systems Operational
            </p>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;