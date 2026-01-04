//
//  APIproduct.swift
//  VitalClient
//
//  Created by Dylan Cairns on 1/3/26.
//
struct APIproduct: Identifiable, Codable {
    let id: Int
    let productName: String
    
    enum CodingKeys: String, CodingKey {
        case id
        case productName = "product_name"
    }
}
