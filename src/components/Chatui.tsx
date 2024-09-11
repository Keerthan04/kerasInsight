import React from 'react';

export default function ChatUI() {
  return (
    <div className="h-screen flex bg-red-500 text-white">
      {/* Sidebar */}
      <div className="w-1/4 bg-red-600">
        <div className="h-full p-4">
          <div className="mb-4 text-xl font-bold">Component List</div>
          <ul className="space-y-2">
            <li>Component A</li>
            <li>Component B</li>
            <li>Component C</li>
          </ul>
        </div>
      </div>
      {/* Main Section */}
      <div className="flex-1 flex flex-col">
        <header className="p-4 bg-red-700 text-lg font-semibold">
          Current Component: Component A
        </header>
        <div className="flex-1 p-4 overflow-y-auto bg-red-400">
          <div className="mb-2 text-sm">10:00 AM - Message Received</div>
          <div className="mb-2 text-sm text-right">10:01 AM - Message Sent</div>
          {/* Conversation History with timestamps */}
          {/* Message Received */}
          <div className="bg-red-900 text-white p-2 my-1 max-w-xs rounded-t-xl rounded-bl-xl">Received: Hi there!</div>
          {/* Message Sent */}
          <div className="bg-red-800 text-white p-2 my-1 max-w-xs self-end rounded-t-xl rounded-bl-xl">Sent: Hello, how can I help you?</div>
        </div>
        <div className="p-4 flex items-center bg-red-700">
          <input type="text" placeholder="Type your message..." className="flex-1 px-2 py-1 rounded-l-lg bg-red-600 text-white"></input>
          <button className="px-4 py-2 bg-red-900 rounded-r-lg">Send</button>
        </div>
      </div>
    </div>
  );
}
