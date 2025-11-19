import React from 'react'
import Nifty50Predictor from './components/Nifty50Predictor'

export default function App(){
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Nifty50Predictor apiBase="http://localhost:8000" />
    </div>
  )
}
