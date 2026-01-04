//
//  ContentView.swift
//  VitalClient
//
//  Created by Dylan Cairns on 1/3/26.
//

import SwiftUI

struct ContentView: View {
    @State private var products: [APIproduct] = []
    
    func testProductFetch() {
        guard let url = URL(string: "http://127.0.0.1:8000/products") else { return }
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let data = data {
                do {
                    let decodedData = try JSONDecoder().decode([APIproduct].self, from: data)
                    print(decodedData)
                    DispatchQueue.main.async {
                        self.products = decodedData
                    }
                } catch {
                    print("Errør:", error)
                }
            } else if let error = error {
                print("Error:", error)
            }
        }.resume()
    }
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            List(products) { product in
                Text(product.productName)
            }
        }
        .padding()
        .onAppear{
            testProductFetch()
        }
    }
}

#Preview {
    ContentView()
}
